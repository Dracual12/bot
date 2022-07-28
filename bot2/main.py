from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
import logging
from aiogram.utils import executor
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import os
import json
import torchvision.transforms as transforms
from translate import Translator
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np

translator = Translator(to_lang="Russian")

inline_btn_1 = InlineKeyboardButton('Сегментация', callback_data='button1')
inline_btn_2 = InlineKeyboardButton('Классификация', callback_data='button2')

inline_kb1 = InlineKeyboardMarkup().add(inline_btn_1)
inline_kb1.add(inline_btn_2)

token = '5468457625:AAFBb0qwtpAMfKqBW2MtY3BFSuaaHHRciNI'
bot = Bot(token=token)
dp = Dispatcher(bot)
logging.basicConfig(level=logging.INFO)

flag1 = False
flag2 = False


def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def xyz():
    img = Image.open('test.jpg')
    plt.imshow(img)
    fcn = models.segmentation.fcn_resnet101(pretrened=True).eval()
    trf = T.Compose([T.Resize(256),
                     T.CenterCrop(224),
                     T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    inp = trf(img).unsqueeze(0)
    out = fcn(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om)
    plt.imshow(rgb)
    plt.savefig('t.jpg')
    path = os.getcwd()
    os.remove(path=path+'/test.jpg')


def get_idx_to_label():
    with open("imagenet_idx_to_label.json") as f:
        return json.load(f)


def get_image_transform():
    transform = transforms.Compose([
      transforms.Resize(224),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform


def predict(image):
    model = models.resnet18(pretrained=True)
    model.eval()
    out = model(image)
    _, pred = torch.max(out, 1)
    idx_to_label = get_idx_to_label()
    cls = idx_to_label[str(int(pred))]
    translation = translator.translate(cls)
    return translation


def load_image():
    image = Image.open('test.jpg')
    transform = get_image_transform()
    image = transform(image)[None]
    return image


@dp.message_handler(commands=['start'])
async def start(msg:types.Message):
    await msg.answer('Вы хотите классифицировать или сегментировать картинку?', reply_markup=inline_kb1)


@dp.callback_query_handler(lambda c: c.data == 'button1')
async def process_callback_button1(callback_query: types.CallbackQuery):
    global flag1
    await bot.answer_callback_query(callback_query.id)
    await bot.delete_message(chat_id=callback_query.from_user.id, message_id=callback_query.message.message_id)
    await bot.send_message(callback_query.from_user.id, 'Отправьте любое изображение')
    flag1 = True


@dp.callback_query_handler(lambda c: c.data == 'button2')
async def process_callback_button1(callback_query: types.CallbackQuery):
    global flag2
    await bot.answer_callback_query(callback_query.id)
    await bot.delete_message(chat_id=callback_query.from_user.id, message_id=callback_query.message.message_id)
    await bot.send_message(callback_query.from_user.id, 'Отправьте фото с животным')
    flag2 = True


@dp.message_handler(content_types=['photo'])
async def get_photo(message: types.Message):
    global flag1, flag2
    if flag1:
        await message.photo[-1].download('test.jpg')
        xyz()
        path = os.getcwd()
        await bot.send_photo(message.chat.id, types.InputFile(path + '/t.jpg'))
        os.remove(path=path + '/t.jpg')
        flag1 = False
    if flag2:
        await message.photo[-1].download('test.jpg')
        path = os.getcwd()
        x = load_image()
        await message.answer(f'Мне кажется это: {predict(x)}')
        os.remove(path=path + '/test.jpg')
        flag2 = False

##@dp.callback_query_handler(func=lambda call: True)

if __name__ == '__main__':
    executor.start_polling(dp)