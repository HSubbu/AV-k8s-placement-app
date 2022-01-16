import requests

candidate = [{"gender": "M",
  "ssc_p": 71.0,
  "ssc_b": 'Central',
  "hsc_p": 58.66,
  "hsc_b": 'Central',
  "hsc_s": 'Science',
  "degree_p": 58.0,
  "degree_t": 'Sci&Tech',
  "etest_p": 56.0,
  "mba_p": 61.3,
  "specialisation": 'Mkt&Fin',
  "workex": 'Yes',
  }]


#url = "http://0.0.0.0:9696/predict"

url = "http://127.0.0.1:55395/predict"
result = requests.post(url=url,json=candidate).json()
print('The Model Prediction for placement :',result)





