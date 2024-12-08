use crate::{layers::HiddenActivation, Batch, ModelType};
use candle_core::{Result, Tensor, Device};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use serde::Deserialize;
use crate::Model;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct BertConfig {
    pub attention_probs_dropout_prob: f64,
    pub bos_token_id: usize,
    pub classifier_dropout: Option<f64>,
    pub eos_token_id: usize,
    pub hidden_act: HiddenActivation,
    pub hidden_dropout_prob: f64,
    pub hidden_size: usize,
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub layer_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub output_past: bool,
    pub pad_token_id: usize,
    #[serde(default)]
    pub position_embedding_type: PositionEmbeddingType,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub type_vocab_size: usize,
    pub use_cache: bool,
    pub vocab_size: usize,
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    #[default]
    Absolute,
}

pub struct BertEmbeddings {
    pub word_embeddings: Embedding,
    pub position_embeddings: Embedding,
    pub token_type_embeddings: Embedding,
    pub layer_norm: LayerNorm,
}

impl BertEmbeddings {
    pub fn load(vb: VarBuilder,config: &BertConfig) -> Result<Self> {
        if config.position_embedding_type != PositionEmbeddingType::Absolute {
            unimplemented!("unsupported position embedding type");
        }

        Ok(Self {
            word_embeddings: Embedding::new(
                vb.pp("word_embeddings")
                    .get((config.vocab_size, config.hidden_size), "weight")?,
                config.hidden_size,
            ),
            position_embeddings: Embedding::new(
                vb.pp("position_embeddings").get(
                    (config.max_position_embeddings, config.hidden_size),
                    "weight",
                )?,
                config.hidden_size,
            ),
            token_type_embeddings: Embedding::new(
                vb.pp("token_type_embeddings")
                    .get((config.type_vocab_size, config.hidden_size), "weight")?,
                config.hidden_size,
            ),
            layer_norm: LayerNorm::new(
                vb.pp("LayerNorm").get(config.hidden_size, "weight")?,
                vb.pp("LayerNorm").get(config.hidden_size, "bias")?,
                config.layer_norm_eps,
            ),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        println!("start embedding");
        println!("input_ids dims: {:?}", input_ids.dims());
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        println!("input_embeddings dims: {:?}", input_embeddings.dims());
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        println!("token_type_embeddings dims: {:?}", token_type_embeddings.dims());
        let position_embeddings = self.position_embeddings.forward(position_ids)?;
        println!("position_embeddings dims: {:?}", position_embeddings.dims());
        let embeddings = input_embeddings
            .add(&token_type_embeddings)?
            .add(&position_embeddings)?;
        let embeddings = self.layer_norm.forward(&embeddings)?;
        println!("embeddings dims: {:?}", embeddings.dims());
        Ok(embeddings)
    }
}

pub struct BertLayer {
    attention: BertAttention,
    intermediate: Linear,
    output: Linear,
    layer_norm: LayerNorm
}

impl BertLayer {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let attention = BertAttention::load(vb.pp("attention"), config)?;
        let intermediate_weight = vb
        .pp("intermediate")
        .pp("dense")
        .get((config.intermediate_size, config.hidden_size), "weight")?;

        let intermediate_bias = vb
        .pp("intermediate")
        .pp("dense")
        .get(config.intermediate_size, "bias")?;

        let intermediate = Linear::new(intermediate_weight, Some(intermediate_bias));

        let output_weight = vb
        .pp("output")
        .pp("dense")
        .get((config.hidden_size, config.intermediate_size), "weight")?;

        let output_bias = vb
        .pp("output")
        .pp("dense")
        .get(config.hidden_size, "bias")?;

        let output = Linear::new(output_weight, Some(output_bias));

        let layer_norm_weight = vb
        .pp("output")
        .pp("LayerNorm")
        .get(config.hidden_size, "weight")?;

        let layer_norm_bias = vb
        .pp("output")
        .pp("LayerNorm")
        .get(config.hidden_size, "bias")?;

        let layer_norm = LayerNorm::new(layer_norm_weight, layer_norm_bias, config.layer_norm_eps);

        Ok(Self { attention, intermediate, output, layer_norm })

    }
}

pub struct BertAttention {
    qkv_linear: Linear,
    dense: Linear,
    layer_norm: LayerNorm,
    softmax_scale: f64
}

impl BertAttention {

    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let hidden_size = config.hidden_size;
        let query_weight = vb.pp("self.query").get((all_head_size, hidden_size), "weight")?;
        let query_bias = vb.pp("self.query").get(all_head_size, "bias")?;

        let key_weight = vb.pp("self.key").get((all_head_size, hidden_size), "weight")?;
        let key_bias = vb.pp("self.key").get(all_head_size, "bias")?;

        let value_weight = vb.pp("self.value").get((all_head_size, hidden_size), "weight")?;
        let value_bias = vb.pp("self.value").get(all_head_size, "bias")?;


        let qkv_weight = Tensor::cat(&[&query_weight, &key_weight, &value_weight], 0)?;
        let qkv_bias = Tensor::cat(&[&query_bias, &key_bias, &value_bias], 0)?;
        let qkv_linear = Linear::new(qkv_weight, Some(qkv_bias));

        let dense_weight = vb.pp("output.dense").get((hidden_size, hidden_size), "weight")?;
        let dense_bias = vb.pp("output.dense").get(hidden_size, "bias")?;

        let dense = Linear::new(dense_weight, Some(dense_bias));
        let layer_norm = LayerNorm::new(
            vb.pp("output.LayerNorm").get(hidden_size, "weight")?, 
            vb.pp("output.LayerNorm").get(hidden_size, "bias")?, config.layer_norm_eps);
        let softmax_scale = 1. / (attention_head_size as f64).sqrt();
        Ok(Self{
            qkv_linear,
            dense,
            layer_norm,
            softmax_scale,
        })
    }
    
    fn forward(&self, hidden_states: &Tensor, attention_bias: Option<&Tensor>) -> Result<Tensor> {
        let qkv = self.qkv_linear.forward(hidden_states)?;
        println!("qkv dims : {:?}", qkv.dims());
        let a = Tensor::new(&[[0.0f32, 1.0, 2.0], [3.0, 4.0, 5.0]], &Device::Cpu)?;
        Ok(a)
    }
}

pub struct BertIntermediate {
}

pub struct BertOutput {
}

pub struct BertEncoder {
    layers: Vec<BertLayer>
}

impl BertEncoder {
    pub fn load(vb: VarBuilder, config: &BertConfig) -> Result<Self> {
        println!("start bert encoder load");
        let layers = (0..config.num_hidden_layers).
        map(|i| BertLayer::load(vb.pp(&format!("layer.{}", i)), config)).collect::<Result<Vec<_>>>()?;
        println!("bert encoder load done");
        Ok(Self { layers })
    }
}

pub struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    num_attention_heads: usize,
    device: Device,
}

impl BertModel {
    pub fn load_roberta(vb: VarBuilder, config: &BertConfig, model_type: ModelType) -> Result<Self> {
        if config.position_embedding_type != PositionEmbeddingType::Absolute {
            unimplemented!("unsupported embedding type")
        }
 
        let (embeddings, encoder) = match (
            BertEmbeddings::load(vb.pp("embeddings"), config),
            BertEncoder::load(vb.pp("encoder"), config)
         ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            _ => unimplemented!("failed to load embeddings or encoder")
        };

        Ok(Self{
            embeddings,
            encoder,
            num_attention_heads: config.num_attention_heads,
            device: vb.device().clone()
        })
    }

    pub fn forward(&self, batch: Batch) -> Result<Tensor> {
        let batch_size = batch.len();
        let max_length = batch.max_length as usize;
        let shape = (batch_size, max_length);

        let input_ids = Tensor::from_vec(batch.input_ids, shape, &self.device)?;
        let type_ids = Tensor::from_vec(batch.token_type_ids, shape, &self.device)?;
        let position_ids = Tensor::from_vec(batch.position_ids, shape, &self.device)?;

        let embedding_output = self.embeddings.forward(&input_ids, &type_ids, &position_ids)?;
        println!("embedding_output dims: {:?}", embedding_output.dims());
        let a = Tensor::new(&[[0.0f32, 1.0, 2.0], [3.0, 4.0, 5.0]], &Device::Cpu)?;
        Ok(a)
    }

}

impl Model for BertModel {
    fn embed(&self, batch: Batch) -> Result<Tensor> {
        println!("start embedding");
        self.forward(batch)
    }
}