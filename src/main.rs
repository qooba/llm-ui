use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use anyhow::Result;
use bytes::{Bytes, BytesMut};
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
use uuid::Uuid;

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

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let (tx_infer, rx_infer) = sync_channel::<String>(3);
    let (tx_callback, rx_callback) = channel::<String>(3);

    TX_INFER.set(Arc::new(Mutex::new(tx_infer))).unwrap();
    RX_CALLBACK.set(Arc::new(Mutex::new(rx_callback))).unwrap();

    thread::spawn(move || {
        let llama = llm::load::<llm::models::Llama>(
            std::path::Path::new("/home/jovyan/rust-src/llm-ui/models/ggml-model-q4_0.bin"),
            Default::default(),
            llm::load_progress_callback_stdout,
        )
        .unwrap_or_else(|err| panic!("Failed to load model: {err}"));

        while let Ok(msg) = rx_infer.recv() {
            let prompt = msg.to_string();
            let mut session = llama.start_session(Default::default());
            let res = session.infer::<std::convert::Infallible>(
                &llama,
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
        }
    });

    HttpServer::new(|| App::new().route("/chat", web::get().to(chat)))
        .bind(("0.0.0.0", 8089))?
        .run()
        .await
}
