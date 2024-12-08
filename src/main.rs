mod download;
mod error;
mod infer;
mod tokenization;

use crate::download::download_artifacts;
use crate::error::TextEmbeddingsError;
use crate::tokenization::{EncodingInput, Tokenization};
use anyhow::Result;
use candle_core::{Device, Tensor};

#[tokio::main]
async fn main() -> Result<(), TextEmbeddingsError> {
    let api = hf_hub::api::tokio::Api::new().unwrap();
    let api_repo = api.model("BAAI/bge-m3".to_string());
    let model_path = download_artifacts(&api_repo).await?;
    println!("model_path: {:?}", model_path);
    // let mut tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))?;
    // let tokenization = Tokenization::new();
    // let encoded = tokenization.encode(&mut tokenizer, EncodingInput::Text("Hello, world!".to_string()) , true, true, 512)?;
    // println!("tokens: {:?}", encoded.get_tokens());
    // println!("ids: {:?}", encoded.get_ids());
    // println!("vocab size: {:?}", tokenizer.get_vocab_size(true));

    // let encoded2 = tokenizer.encode("你好， 很高兴见到你", true)?;
    // println!("tokens: {:?}", encoded2.get_tokens());
    // println!("ids: {:?}", encoded2.get_ids());

    // let decoded = tokenization.decode_ids(&mut tokenizer, encoded2.get_ids().to_vec(), true)?;
    // println!("decoded: {:?}", decoded);
    // let a = Tensor::arange(0f32, 6f32, &Device::Cpu)
    //     .unwrap()
    //     .reshape((2, 3))
    //     .unwrap();
    // let tensors: std::collections::HashMap<String, Tensor> =
    //     [("foo".to_string(), a)].into_iter().collect();

    // let vb = candle_nn::VarBuilder::from_tensors(tensors, candle_core::DType::F32, &Device::Cpu);
    // println!("prefix: {}", vb.get((2, 3), "foo").unwrap());
    Ok(())
}
