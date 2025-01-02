import torch
import argparse
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write

def parse_args():
    parser = argparse.ArgumentParser(description="Process input text and output directory path.")
    parser.add_argument('--text', type=str, required=True, help='Input text to synthesize')
    parser.add_argument('--output', type=str, required=True, help='Output WAV file path')
    return parser.parse_args()

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

CONFIG_PATH = "./configs/vits2_ljs_ring.json"
MODEL_PATH = "./logs/G_100000.pth"

if __name__ == "__main__":
    args = parse_args()

    hps = utils.get_hparams_from_file(CONFIG_PATH)

    if (
        "use_mel_posterior_encoder" in hps.model.keys()
        and hps.model.use_mel_posterior_encoder == True
    ):
        print("Using mel posterior encoder for VITS2")
        posterior_channels = 80  # vits2
        hps.data.use_mel_posterior_encoder = True
    else:
        print("Using lin posterior encoder for VITS1")
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    ).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint(MODEL_PATH, net_g, None)

    text = args.text
    stn_tst = get_text(text, hps)

    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()

        audio = (
            net_g.infer(
                x_tst, x_tst_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )

        write(data=audio, rate=hps.data.sampling_rate, filename=args.output)
        print(f"Audio file saved to {args.output}")
