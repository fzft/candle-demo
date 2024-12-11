mod compute_cap;
mod layers;
mod models;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use serde::Deserialize;
use std::{path::Path, vec};
use nohash_hasher::IntMap;
use std::collections::HashMap;
use nohash_hasher::BuildNoHashHasher;

use models::{BertConfig, BertModel, Model};

#[derive(Debug)]
pub struct Batch {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub cumulative_seq_lengths: Vec<u32>,
    pub position_ids: Vec<u32>,
    pub max_length: u32,
    pub pooled_indices: Vec<u32>
}

impl Batch {
    pub fn len(&self) -> usize {
        self.cumulative_seq_lengths.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum ModelType {
    Embedding(Pool)
}

#[derive(Debug, PartialEq, Clone)]
pub enum Pool {
    Cls 
}

#[derive(Deserialize, Debug)]
#[serde(tag = "model_type", rename_all = "kebab-case")]
enum Config {
    XlmRoberta(BertConfig),
}

pub struct Backend {
    device: Device,
    model: Box<dyn Model>
}

pub enum Embedding {
    Pooled(Vec<f32>),
    All(Vec<Vec<f32>>),
}

pub type Embeddings = IntMap<usize, Embedding>;

impl Backend {
    pub fn new(model_path: &Path, model_type: ModelType) -> Result<Self, anyhow::Error> {
        let default_safetensors = model_path.join("model.safetensors");
        let default_pytorch = model_path.join("pytorch_model.bin");

        let model_files = if default_safetensors.exists() {
            vec![default_safetensors]
        } else if default_pytorch.exists() {
            vec![default_pytorch]
        } else {
            // Sharede weights
            todo!()
        };

        let config = std::fs::read_to_string(model_path.join("config.json")).unwrap();
        println!("config: {:?}", config);
        let config: Result<Config, serde_json::Error> = serde_json::from_str(&config);

        let config = match config {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!("Failed to parse config.json: {:?}", e);
                return Err(anyhow::anyhow!("Invalid config file: {:?}", e));
            }
        };

        let device = if candle_core::utils::cuda_is_available() {
            println!("use cuda devide");
            Device::new_cuda(0)
        } else {
            println!("use cpu device");
            Ok(Device::Cpu)
        }?;

        let dtype = DType::F16;

        let vb = if model_files.len() == 1 && model_files[0].extension().unwrap() == "bin" {
            VarBuilder::from_pth(&model_files[0], dtype, &device)
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&model_files, dtype, &device) }
        }?;

        let model = match (config, &device) {
            (Config::XlmRoberta(config), Device::Cuda(_)) => {
                let model = BertModel::load_roberta(vb, &config, model_type)?;
                Box::new(model) as Box<dyn Model>
            }
            _ => {
                unimplemented!("unsupported model type")
            }
        };

        Ok(Self { device, model })
    }


    pub fn embed(&self, batch: Batch) -> Result<Embeddings, anyhow::Error> {
        let batch_size = batch.len();
        let pooled_indices = batch.pooled_indices.clone();
        let (pooled_embeddings, _) = self.model.embed(batch).unwrap();
        let pooled_embeddings = match pooled_embeddings {
            None => vec![],
            Some(pooled_embeddings) => pooled_embeddings.to_dtype(DType::F32).unwrap().to_vec2().unwrap()
        };

        let mut embeddings =
        HashMap::with_capacity_and_hasher(batch_size, BuildNoHashHasher::default());
        for (i, e) in pooled_indices.into_iter().zip(pooled_embeddings) {
            embeddings.insert(i as usize, Embedding::Pooled(e));
        }

        Ok(embeddings)
        
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use tokenizers::Tokenizer;

    use crate::Embedding;

    use super::{Backend, Batch, ModelType, Pool};

    #[test]
    fn test_backend() {
        let model_path = Path::new("/home/fzft/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181");
        let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json")).unwrap();
        let encoding = tokenizer.encode("你好， 很高兴见到你", true).unwrap();
        let model = Backend::new(model_path, ModelType::Embedding(Pool::Cls)).unwrap();
        let seq_len = encoding.len();
        let position_offset = 0;
        let entry_tokens =  encoding.get_ids().len();
        let cu_seq_lengths = vec![0, entry_tokens as u32];
        let batch = Batch {
            input_ids: encoding.get_ids().to_vec(),
            token_type_ids: encoding.get_type_ids().to_vec(),
            cumulative_seq_lengths: cu_seq_lengths,
            position_ids: (position_offset as u32..(seq_len + position_offset) as u32)
            .collect::<Vec<_>>(),
            max_length: encoding.get_ids().to_vec().len() as u32,
            pooled_indices: vec![]
        };
        println!("batch: {:?}", batch);
        let embeddings = model.embed(batch).unwrap();
        for  v in embeddings.into_values() {
            match v {
                Embedding::Pooled(e) => {
                    println!("embedding: {:?}", e);
                }
                Embedding::All(e) => {
                    println!("embedding: {:?}", e);
                }
            }
        }
    }
}
