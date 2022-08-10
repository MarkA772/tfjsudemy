const HtmlWebpackPlugin = require("html-webpack-plugin");

module.exports = {
  mode: "development",
  plugins: [
    new HtmlWebpackPlugin({
      template: "./index.html"
    }),
  ],
  devServer: {
    static: ['static']
  },
  devtool: 'eval-cheap-source-map',
};