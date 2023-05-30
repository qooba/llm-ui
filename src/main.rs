use actix_files as fs;
use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use anyhow::Result;
use bytes::{Bytes, BytesMut};
use clap::{Parser, Subcommand, ValueEnum};
use llm::Model;
use once_cell::sync::{Lazy, OnceCell};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::Arc;
use std::thread;
use std::{convert::Infallible, io::Write, path::PathBuf};
use tokio::sync::mpsc::{channel, Receiver};
use tokio::sync::Mutex;

#[derive(Parser, Debug, Clone)]
struct Args {
    model_architecture: llm::ModelArchitecture,
    model_path: PathBuf,
    #[arg(long, short = 'v')]
    vocabulary_path: Option<PathBuf>,
    #[arg(long, short = 'r')]
    vocabulary_repository: Option<String>,
    #[arg(long, short = 'h')]
    host: String,
    #[arg(long, short = 'p')]
    port: u16,
}
impl Args {
    pub fn to_vocabulary_source(&self) -> llm::VocabularySource {
        match (&self.vocabulary_path, &self.vocabulary_repository) {
            (Some(_), Some(_)) => {
                panic!("Cannot specify both --vocabulary-path and --vocabulary-repository");
            }
            (Some(path), None) => llm::VocabularySource::HuggingFaceTokenizerFile(path.to_owned()),
            (None, Some(repo)) => llm::VocabularySource::HuggingFaceRemote(repo.to_owned()),
            (None, None) => llm::VocabularySource::Model,
        }
    }
}

static TX_INFER: OnceCell<Arc<Mutex<SyncSender<String>>>> = OnceCell::new();
static RX_CALLBACK: OnceCell<Arc<Mutex<Receiver<llm::InferenceResponse>>>> = OnceCell::new();

#[derive(Deserialize, Debug, Clone)]
pub struct ChatRequest {
    pub prompt: String,
}

async fn chat(chat_request: web::Query<ChatRequest>) -> Result<impl Responder, Box<dyn Error>> {
    TX_INFER
        .get()
        .unwrap()
        .lock()
        .await
        .send(chat_request.prompt.to_string());

    let mut rx_callback = RX_CALLBACK.get().unwrap().lock().await;
    let stream_tasks = async_stream::stream! {
        let mut bytes = BytesMut::new();
        while let Some(msg) = &rx_callback.recv().await {

            match msg {
                llm::InferenceResponse::PromptToken(t) | llm::InferenceResponse::InferredToken(t) => {
                    let b = t.to_string().as_bytes();
                    bytes.extend_from_slice(t.as_bytes());
                    let byte = bytes.split().freeze();
                    yield Ok::<Bytes, Box<dyn Error>>(byte);

                }
                _ => break

            }
        }
    };

    Ok(HttpResponse::Ok()
        .content_type("application/json")
        .streaming(Box::pin(stream_tasks)))
}

fn infer(
    args: &Args,
    rx_infer: std::sync::mpsc::Receiver<String>,
    tx_callback: tokio::sync::mpsc::Sender<llm::InferenceResponse>,
) -> Result<()> {
    let vocabulary_source = args.to_vocabulary_source();
    let model_architecture = args.model_architecture;
    let model_path = &args.model_path;
    let now = std::time::Instant::now();

    let llm_model = llm::load_dynamic(
        model_architecture,
        &model_path,
        vocabulary_source,
        Default::default(),
        llm::load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| {
        panic!("Failed to load {model_architecture} model from {model_path:?}: {err}")
    });

    println!(
        "Model fully loaded! Elapsed: {}ms",
        now.elapsed().as_millis()
    );

    while let Ok(msg) = rx_infer.recv() {
        let prompt = msg.to_string();
        let mut session = llm_model.start_session(Default::default());

        let res = session.infer::<std::convert::Infallible>(
            llm_model.as_ref(),
            &mut rand::thread_rng(),
            &llm::InferenceRequest {
                prompt: Some(prompt).as_deref().unwrap().into(),
                parameters: &llm::InferenceParameters::default(),
                play_back_previous_tokens: false,
                maximum_token_count: None,
            },
            &mut Default::default(),
            |r| {
                tx_callback.blocking_send(r);
                Ok(llm::InferenceFeedback::Continue)
            },
        );

        tx_callback.blocking_send(llm::InferenceResponse::EotToken);
        println!("INFER END");
    }

    Ok(())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();

    let (tx_infer, rx_infer) = sync_channel::<String>(3);
    let (tx_callback, rx_callback) = channel::<llm::InferenceResponse>(3);

    TX_INFER.set(Arc::new(Mutex::new(tx_infer))).unwrap();
    RX_CALLBACK.set(Arc::new(Mutex::new(rx_callback))).unwrap();

    let host = args.host.to_string();
    let port: u16 = args.port.clone();

    thread::spawn(move || {
        infer(&args, rx_infer, tx_callback);
    });


    HttpServer::new(|| {
        App::new()
            .route("/api/chat", web::get().to(chat))
            .service(fs::Files::new(
                "/",
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("static"),
            ))
    })
    .bind((host, port))?
    .run()
    .await
}
