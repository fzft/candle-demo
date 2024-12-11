mod bert;

pub use bert::{BertConfig, BertModel};

use candle_core::{Result, Tensor};
use crate::Batch;

pub(crate) trait Model {
    fn embed(&self, _batch: Batch) -> Result<(Option<Tensor>, Option<Tensor>)> {
        todo!()
    }
}