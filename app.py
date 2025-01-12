from flask import Flask, jsonify, request
from chatbot import make_prompt, evaluate_response, output_audio, input_audio, init_response

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Welcome to the game API"}), 200

@app.route("/chat", methods=["POST"])
currently_playing = True
difficulty = 1

def main():
    while True:
        init_response()
        user_input = input_audio() 

        if user_input:
            conversation_type = user_input
            if conversation_type in ["professional", "romantic", "romantics", "casual"]:
                prompt = make_prompt(conversation_type, difficulty)
                output_audio(prompt)

                            
                response = input_audio()  
                
                evaluation = evaluate_response(prompt, response)
                output_audio(evaluation)

                output_audio("Would you like to continue to the next level?")

                intent = input_audio()
                if intent == 'Yes':
                    currently_playing = True
                    difficulty += 1
            
            else:
                output_audio("Invalid conversation type. Please choose Professional, Romantic, or Casual.")
        else:s
            output_audio("Sorry, I didn't catch that. Please say again.")


if __name__ == '__main__':
    app.run(port=5010)
