var webpack = require('webpack');
const path = require('path');
const buildDirectoryCAPM = 'dapi/capm/static/capm/public/compiled';
const filename = 'bundle.js';

module.exports = {
  entry: {
    landing: ['./ui/entry.js']
  },
  output: {
    path: path.join(__dirname, buildDirectoryCAPM),
    filename: '[name].js'
  },
  module: {
    rules: [
        {
            test: /\.js$/,
            exclude: /node_modules/,
            use: {
                loader: 'babel-loader'
            }
        }
    ]
  },
  watchOptions: {
    aggregateTimeout: 100,
    ignored: '/node_modules'
  }
}
