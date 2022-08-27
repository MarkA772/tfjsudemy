const io = require("socket.io");

const server = new io.Server({
  cors: {
    origin: "http://localhost:8080"
  }
});

server.on("connection", (socket) => {
  console.log("user connected");

  socket.on('test', (data) => {
    const d = JSON.parse(data);
    console.log(d.config.layers);
  });
  
});

server.listen(3333);


