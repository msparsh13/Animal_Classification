from flask import Flask, request, render_template
import pickle
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from tensorflow.keras.utils import load_img , img_to_array
from tensorflow import keras
app = Flask(__name__ , template_folder="templates")



model = load_model('models/AC.h5' , compile=False)


lab = {0:'antelope',
    1:'badger',
 2:'bat',
 3:'bear',
 4:'bee',
 5:'beetle',
 6:'bison',
 7:'boar',
 8:'butterfly',
 9:'cat',
 10:'caterpillar',
 11:'chimpanzee',
 12:'cockroach',
 13:'cow',
 14:'coyote',
 15:'crab',
 16:'crow',
 17:'deer',
 18:'dog',
 19:'dolphin',
 20:'donkey',
 21:'dragonfly',
 22:'duck',
 23:'eagle',
 24:'elephant',
 25:'flamingo',
 26:'fly',
 27:'fox',
 28:'goat',
 29:'goldfish',
 30:'goose',
 31:'gorilla',
 32:'grasshopper',
 33:'hamster',
 34:'hare',
 35:'hedgehog',
 36:'hippopotamus',
 37:'hornbill',
 38:'horse',
 39:'hummingbird',
 40:'hyena',
 41:'jellyfish',
 42:'kangaroo',
 43:'koala',
 44:'ladybugs',
 45:'leopard',
 46:'lion',
 47:'lizard',
 48:'lobster',
 49:'mosquito',
 50:'moth',
 51:'mouse',
 52:'octopus',
 53:'okapi',
 54:'orangutan',
 55:'otter',
 56:'owl',
 57:'ox',
 58:'oyster',
 59:'panda',
 60:'parrot',
 61:'pelecaniformes',
 62:'penguin',
 63:'pig',
 64:'pigeon',
 65:'porcupine',
 66:'possum',
 67:'raccoon',
 68:'rat',
 69:'reindeer',
 70:'rhinoceros',
 71:'sandpiper',
 72:'seahorse',
 73:'seal',
 74:'shark',
 75:'sheep',
 76:'snake',
 77:'sparrow',
 78:'squid',
 79:'squirrel',
 80:'starfish',
 81:'swan',
 82:'tiger',
 83:'turkey',
 84:'turtle',
 85:'whale',
 86:'wolf',
 87:'wombat',
 88:'woodpecker',
 89:'zebra'}
def output(location):
    img = load_img(location , target_size=(200, 200 , 3))
    img = img_to_array(img)
    #img= img/255
    
    img= np.expand_dims(img , [0])
    print(img.shape)
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    return res

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/submit',methods=['POST'])
def get():
    
    try:
        img = request.files['my_image']
        img_path = "static/"+img.filename
        img.save(img_path)
        p = output(img_path)

        return render_template("index.html" , prediction=p , img_path = img_path)
    except:
        print("error")
        return render_template("index.html") 

 

if __name__ == "__main__":
    app.run(debug=True)