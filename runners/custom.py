import os
import sys
import lightning.pytorch as pl
import hydra

sys.path.append(os.getcwd())
from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402
from ocr.lightning_modules.got_ocr2 import GOT_OCR2_BBox_LitModule

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


from ocr.models import get_model_by_cfg
from ocr.datasets import get_datasets_by_cfg
from ocr.lightning_modules.ocr_pl import OCRPLModule, OCRDataPLModule


def get_data_modules(config):
    dataset = get_datasets_by_cfg(config.datasets)
    data_modules = OCRDataPLModule(dataset=dataset, config=config)
    return data_modules




@hydra.main(config_path=CONFIG_DIR, config_name='predict', version_base='1.2')
def predict(config):
    """
    Train a OCR model using the provided configuration.

    Args:
        `config` (dict): A dictionary containing configuration settings for predict.
    """
    pl.seed_everything(config.get("seed", 42), workers=True)

    model_module = GOT_OCR2_BBox_LitModule()

    data_module = get_data_modules(config)

    trainer = pl.Trainer(logger=False)

    ckpt_path = config.get("checkpoint_path")
    assert ckpt_path, "checkpoint_path must be provided for prediction"

    trainer.predict(model_module,
                    data_module,
                    ckpt_path=ckpt_path,
                    )

if __name__ == "__main__":
    predict()
