from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

#this endpoint returns the index html file to format requests with javascript
@app.route('/')
def home():
    return render_template('index.html')

stores = [
    {
    'name':'My wonderful store',
        'items':[
            {
            'name':'My Item',
            'price':15.99
            }
        ]
    }
]

# POST used to receive data (in server)
# GET used to send data back (in server)
# From browser perspective this is the opposite
# But here we are the server not the browser

#POST /store data: {name:}
@app.route('/store', methods=['POST'])
#defining endpoint to be/store and accessible only via POST
def create_store():
    request_data = request.get_json()
    new_store = {
        'name':request_data['name'],
        'items':[]
    }
    stores.append(new_store)
    return jsonify(new_store)

#GET /store/<string:name>
@app.route('/store/<string:name>') #http://127.0.0.1:5000/store/some_name
def get_store(name):
    for store in stores:
        if store['name'] == name:
            return jsonify(store)
        return jsonify({'message':'store not found'})

#GET /store
@app.route('/store')
def get_stores():
    return jsonify({'stores':stores})

#POST /store/<string:name>/item {name:,price}
@app.route('/store/<string:name>/item', methods=['POST'])
def create_item_in_store(name):
    request_data = request_data.get_json()
    for store in stores:
        if store['name']==name:
            new_item = {
                'name': request_data['name'],
                'price':request_data['price']
            }
            store['items'].append(new_item)
            return jsonify(new_item)
    return jsonify({'message':'store not found'})

#GET /store/<string:name>/item
@app.route('/store/<string:name>/item')
def get_items_in_store(name):
    for store in stores:
        if store['name'] == name:
            return jsonify({'items':store['items']})
        return jsonify({'message':'store not found'})


app.run(port=5000)
