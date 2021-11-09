from flask import Flask, render_template, request, jsonify, make_response, send_file
from werkzeug.exceptions import HTTPException
from application import calcepi

from flask import request, session, abort, flash
import os
import folium

server = Flask(__name__)

@server.route("/")
def index():
    
    return 


calcepidash = calcepi.ce_dash(server, '/calcepi/')

if __name__ == "__main__":
        #server.register_error_handler(404, page_not_found)
        #server.secret_key = os.urandom(12)
        server.run()