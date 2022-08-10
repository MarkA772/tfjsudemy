const express = require('express');

const app = express();

app.get('/', (req, res) => {
  return res.status(200).sendFile(__dirname + '/index.html');
})

app.get('/index.js', (req, res) => {
  return res.status(200).sendFile(__dirname + '/dist/main.js');
})

app.listen(3000);