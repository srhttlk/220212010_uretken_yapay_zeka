from flask import Flask, render_template, request
from scenario_enricher import enrich_scenario_local as enrich_scenario
from generator_infer import generate_fake_map_image
import os
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    scenario_text = ""
    enriched_result = ""
    image_filename = ""  # 🔧 BU SATIR GET DURUMU İÇİN KRİTİK

    if request.method == 'POST':
        scenario_text = request.form['scenario']
        enriched_result = enrich_scenario(scenario_text)

        image_path = generate_fake_map_image()  # 'static/generated_xxxx.png'
        image_filename = generate_fake_map_image()  # sadece dosya ismi geliyor artık

    return render_template('index.html',
                        scenario=scenario_text,
                        enriched=enriched_result,
                        image_filename=image_filename)



if __name__ == '__main__':
    app.run(debug=True)
