import requests
import os
from dotenv import load_dotenv
from typing import List, Dict, Text

load_dotenv()


def mapToDict(dat: List, name: Text = 'data', col_iter: int = 0) -> Dict:
    return {dat[col_iter][name][i][0]: dat[col_iter][name][i][1] for i in range(0, len(dat[col_iter][name]))}


class EDSsession():
    # init function
    def __init__(self):
        eds_email = os.getenv('EDS_API_EMAIL', 'secret')
        eds_pass = os.getenv('EDS_API_PASSWORD', 'secret')
        self.session = requests.Session()
        self.payload = {'email': eds_email, 'password': eds_pass}
        self.session.post(
            'https://economicdatasciences.com/login', data=self.payload)

    # benchmark
    def get_benchmark(self, code):
        get_str = 'https://economicdatasciences.com/benchmarks/' + code
        res = self.session.get(get_str)
        return(res.json())

    # funds
    def get_fund(self, code):
        get_str = 'https://economicdatasciences.com/funds/' + code
        res = self.session.get(get_str)
        return(res.json())

    # get base factors
    def get_factors(self):
        get_str = 'https://economicdatasciences.com/factors'
        res = self.session.get(get_str)
        return(res.json())

    # get base factors
    def get_factormap(self):
        get_str = 'https://economicdatasciences.com/factormap'
        res = self.session.get(get_str)
        return(res.json())
