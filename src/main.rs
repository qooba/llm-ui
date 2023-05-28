use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use bytes::{Bytes, BytesMut};
use llm::Model;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::io::Write;
use std::sync::mpsc::sync_channel;
use std::thread;

static LLM: OnceCell<llm::models::Llama> = OnceCell::new();

#[derive(Deserialize, Debug, Clone)]
pub struct InferRequest {
    pub prompt: String,
}

async fn infer(infer_request: web::Query<InferRequest>) -> Result<impl Responder, Box<dyn Error>> {
    let (tx, rx) = sync_channel(3);
    thread::spawn(move || {
        let llama = llm::load::<llm::models::Llama>(
            std::path::Path::new("/home/jovyan/rust-src/llm-ui/models/ggml-model-q4_0.bin"),
            Default::default(),
            llm::load_progress_callback_stdout,
        )
        .unwrap_or_else(|err| panic!("Failed to load model: {err}"));

        //let llama = LLM.get().unwrap();

        let mut session = llama.start_session(Default::default());
        let res = session.infer::<std::convert::Infallible>(
            &llama,
            &mut rand::thread_rng(),
            &llm::InferenceRequest {
                prompt: &infer_request.prompt,
                ..Default::default()
            },
            &mut Default::default(),
            |t| {
                tx.send(t.to_string());
                Ok(())
            },
        );
    });

    let stream_tasks = async_stream::stream! {
        let mut bytes = BytesMut::new();
        while let Ok(msg) = rx.recv() {
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
    /*
    let llm_model = llm::load::<llm::models::Llama>(
        std::path::Path::new("/home/jovyan/rust-src/llm-ui/models/ggml-model-q4_0.bin"),
        Default::default(),
        llm::load_progress_callback_stdout,
    )
    .unwrap_or_else(|err| panic!("Failed to load model: {err}"));

    LLM.set(llm_model)
        .unwrap_or_else(|err| panic!("Failed to set model"));
    */



    HttpServer::new(|| App::new().route("/infer", web::get().to(infer)))
        .bind(("0.0.0.0", 8089))?
        .run()
        .await
}
