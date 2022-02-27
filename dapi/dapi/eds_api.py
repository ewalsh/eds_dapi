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

    # general data function -- personally, I don't love this although
    # it is less code. Simply, I won't always remember the url setup,
    # so I prefer functions with clear, consistent names
    def get_eds_data(self, *args):
        url_str = 'https://economicdatasciences.com'
        assert(len(args) > 0), "Must include url arguments"
        for s in args:
            url_str = url_str + '/' + s

        res = self.session.get(url_str)
        return(res.json())

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

    # get asset holdings for a benchmark
    def get_benchmark_assets_by_class(self, asset_class):
        get_str = 'https://economicdatasciences.com/assets-by-class/' + asset_class
        res = self.session.get(get_str)
        return(res.json())
