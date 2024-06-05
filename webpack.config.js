const path = require('path');

module.exports = {
  // Entry point of the application
  entry: './Users/engyamr/Downloads/Python/script.js',
  
  // Output configuration
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'bundle.js',
  },
  
  // Rules for processing different types of files
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env'],
          },
        },
      },
      // Add more rules for CSS, images, etc. if needed
    ],
  },
  
  // Plugins for additional functionality
  // Add plugins as needed for features like minification, code splitting, etc.
  
  // Development or production mode configuration
  mode: 'development',
  
  // Devtool configuration for source maps
  devtool: 'inline-source-map',
};
