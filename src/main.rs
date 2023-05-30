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
use std::io::Write;
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::Arc;
use std::thread;
use tokio::sync::mpsc::{channel, Receiver};
use tokio::sync::Mutex;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub enum Args {
    /// Use a LLaMA model
    #[command()]
    Llama(Box<BaseArgs>),
}

#[derive(Parser, Debug, Clone)]
pub struct BaseArgs {
    #[arg(short, long)]
    model: String,

    #[arg(short, long)]
    host: String,

    #[arg(short, long)]
    port: u16,
}

const END: &str = "<<END>>";
static TX_INFER: OnceCell<Arc<Mutex<SyncSender<String>>>> = OnceCell::new();

static RX_CALLBACK: OnceCell<Arc<Mutex<Receiver<String>>>> = OnceCell::new();

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
            if msg.to_string() == END {
                break
            }
            let b = msg.to_string().as_bytes();
            bytes.extend_from_slice(msg.as_bytes());
            let byte = bytes.split().freeze();
            yield Ok::<Bytes, Box<dyn Error>>(byte);
        }
    };

    Ok(HttpResponse::Ok()
        .content_type("application/json")
        .streaming(Box::pin(stream_tasks)))
}

fn infer<M: llm::KnownModel + 'static>(
    args: &BaseArgs,
    rx_infer: std::sync::mpsc::Receiver<String>,
    tx_callback: tokio::sync::mpsc::Sender<String>,
) -> Result<()> {
    let llm_model = llm::load::<llm::models::Llama>(
        std::path::Path::new(&args.model),
        Default::default(),
        llm::load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| panic!("Failed to load model: {err}"));

    while let Ok(msg) = rx_infer.recv() {
        let prompt = msg.to_string();
        let mut session = llm_model.start_session(Default::default());
        let res = session.infer::<std::convert::Infallible>(
            &llm_model,
            &mut rand::thread_rng(),
            &llm::InferenceRequest {
                prompt: &prompt,
                play_back_previous_tokens: false,
                ..Default::default()
            },
            &mut Default::default(),
            |t| {
                tx_callback.blocking_send(t.to_string());
                Ok(())
            },
        );

        tx_callback.blocking_send(END.to_string());
        println!("INFER END");
    }

    Ok(())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let cli_args = Args::parse();
    println!("{cli_args:#?}");

    let (tx_infer, rx_infer) = sync_channel::<String>(3);
    let (tx_callback, rx_callback) = channel::<String>(3);

    TX_INFER.set(Arc::new(Mutex::new(tx_infer))).unwrap();
    RX_CALLBACK.set(Arc::new(Mutex::new(rx_callback))).unwrap();

    //"/home/jovyan/rust-src/llm-ui/models/ggml-model-q4_0.binA
    let c_args = cli_args.clone();
    thread::spawn(move || {
        match &cli_args {
            Args::Llama(args) => {
                infer::<llm::models::Llama>(&args, rx_infer, tx_callback);
            }
        };
    });

    let (host, port) = match &c_args {
        Args::Llama(args) => (args.host.to_string(), args.port),
    };

    HttpServer::new(|| {
        App::new()
            .route("/api/chat", web::get().to(chat))
            .service(fs::Files::new("/", std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("static")))
    })
    .bind((host, port))?
    .run()
    .await
}
