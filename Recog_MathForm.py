import cv2
import pandas.io.clipboard as clipboard
from PIL import ImageGrab
from PIL import Image
import os
import sys
import argparse
import logging
import yaml
import re
import albumentations as alb
from albumentations.pytorch import ToTensorV2

import numpy as np
import torch
from torchvision import transforms
from munch import Munch
from transformers import PreTrainedTokenizerFast
from timm.models.layers import StdConv2dSame
from models import get_model

test_transform = alb.Compose(
    [
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        # alb.Sharpen()
        ToTensorV2(),
    ]
)
def parse_args(args, **kwargs):
    args = Munch({'epoch': 0}, **args)
    kwargs = Munch({'no_cuda': False, 'debug': False}, **kwargs)
    args.wandb = not kwargs.debug and not args.debug
    args.device = 'cpu' 
    args.max_dimensions = [args.max_width, args.max_height]
    args.min_dimensions = [args.get('min_width', 32), args.get('min_height', 32)]
    if 'decoder_args' not in args or args.decoder_args is None:
        args.decoder_args = {}
    if 'model_path' in args:
        args.out_path = os.path.join(args.model_path, args.name)
        os.makedirs(args.out_path, exist_ok=True)
    return args



def token2str(tokens, tokenizer):
    if len(tokens.shape) == 1:
        tokens = tokens[None, :]
    dec = [tokenizer.decode(tok) for tok in tokens]
    return [''.join(detok.split(' ')).replace('Ġ', ' ').replace('[EOS]', '').replace('[BOS]', '').replace('[PAD]', '').strip() for detok in dec]


def pad(img: Image, divable=32):
    """Pad an Image to the next full divisible value of `divable`. Also normalizes the image and invert if needed.

    Args:
        img (PIL.Image): input image
        divable (int, optional): . Defaults to 32.

    Returns:
        PIL.Image
    """
    data = np.array(img.convert('LA'))
    data = (data-data.min())/(data.max()-data.min())*255
    if data[..., 0].mean() > 128:
        gray = 255*(data[..., 0] < 128).astype(np.uint8)  # To invert the text to white
    else:
        gray = 255*(data[..., 0] > 128).astype(np.uint8)
        data[..., 0] = 255-data[..., 0]

    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = data[b:b+h, a:a+w]
    if rect[..., -1].var() == 0:
        im = Image.fromarray((rect[..., 0]).astype(np.uint8)).convert('L')
    else:
        im = Image.fromarray((255-rect[..., -1]).astype(np.uint8)).convert('L')
    dims = []
    for x in [w, h]:
        div, mod = divmod(x, divable)
        dims.append(divable*(div + (1 if mod > 0 else 0)))
    padded = Image.new('L', dims, 255)
    padded.paste(im, im.getbbox())
    return padded


def post_process(s: str):
    """Remove unnecessary whitespace from LaTeX code.

    Args:
        s (str): Input string

    Returns:
        str: Processed image
    """
    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\W_^\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', s)
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
        news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
        if news == s:
            break
    return s


def minmax_size(img, max_dimensions=None, min_dimensions=None):
    if max_dimensions is not None:
        ratios = [a/b for a, b in zip(img.size, max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = np.array(img.size)//max(ratios)
            img = img.resize(size.astype(int), Image.BILINEAR)
    if min_dimensions is not None:
        if any([s < min_dimensions[i] for i, s in enumerate(img.size)]):
            padded_im = Image.new('L', min_dimensions, 255)
            padded_im.paste(img, img.getbbox())
            img = padded_im
    return img


def initialize(arguments=None):
    if arguments is None:
        arguments = Munch({'config': 'Models/config.yaml', 'checkpoint': 'Models/MathRecog.pth', 'no_cuda': True, 'no_resize': False})
    logging.getLogger().setLevel(logging.FATAL)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    with open(arguments.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    args = parse_args(Munch(params))
    args.update(**vars(arguments))
    args.wandb = False
    args.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

    model = get_model(args)
#   model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device),strict=False)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    return args, model, tokenizer


def call_model(args, model, tokenizer, img=None):
    encoder, decoder = model.encoder, model.decoder
    img = minmax_size(pad(img), args.max_dimensions, args.min_dimensions)
    img = np.array(pad(img).convert('RGB'))
    t = test_transform(image=img)['image'][:1].unsqueeze(0)
    im = t.to("cpu")

    with torch.no_grad():
        model.eval()
        device = args.device
        encoded = encoder(im.to(device))
        dec = decoder.generate(torch.LongTensor([args.bos_token])[:, None].to(device), args.max_seq_len,
                               eos_token=args.eos_token, context=encoded.detach(), temperature=args.get('temperature', .25))
        pred = post_process(token2str(dec, tokenizer)[0])
    return pred

