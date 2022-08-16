const ws = require('ws');

const wss = new ws.WebSocketServer({
  port: 80
});

wss.on('connection', (ws) => {
  console.log('hi');

  ws.on('message', (data) => {
    console.log(data.toString());
  });

});

