import tempfile
from os import path

from model import CSRNet
import torch
from image import *
import PIL.Image as Image
from flask import jsonify
from flask import json
from flask import request
from flask import Flask
import logging
app = Flask(__name__)


from torchvision import datasets, transforms
transform=transforms.Compose([
                      transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                  ])


                      
model = CSRNet()
#model.cuda()


@app.errorhandler(400)
def bad_request(e):
    return jsonify({"status": "not ok", "message": "this server could not understand your request"})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "not found", "message": "route not found"})


@app.errorhandler(500)
def not_found(e):
    return jsonify({"status": "internal error", "message": "internal error occurred in server"})



@app.route('/detect', methods=['POST'])
def postimage():
    file = request.files.get('upload')
    filename, ext = os.path.splitext(file.filename)
    if ext not in ('.png', '.jpg', '.jpeg'):
         return 'File extension not allowed.'
        # loading the trained weights
    checkpoint = torch.load('0model_best.pth.tar',map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    tmp = tempfile.TemporaryDirectory()

    temp_storage = path.join(tmp.name, file.filename)
    file.save(temp_storage)
    img = transform(Image.open(temp_storage).convert('RGB'))#.cuda()
    output = model(img.unsqueeze(0))

    print("Predicted Count : ", int(output.detach().cpu().sum().numpy()))
    return jsonify(int(output.detach().cpu().sum().numpy()))



if __name__ == "__main__":
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    print ("Starting server on http://localhost:5000")
    print ("Serving ...",  app.run(host='0.0.0.0'))
    print ("Finished !")
    print ("Done !")