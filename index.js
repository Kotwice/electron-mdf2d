const {app, BrowserWindow} = require('electron')

const io = require('socket.io')(4000)

const fs = require('fs')

var spawn = require("child_process").spawn

io.on('connection', client => {

	client.on('calculate', (event) => {
		let tast = spawn('python', ["./resources/scripts/calculate_task.py", JSON.stringify(event)])
	})

	client.on('figure-process', data => {
		client.broadcast.emit('plot-figure-process', JSON.parse(data))
	})

	client.on('figure-result', data => {
		client.broadcast.emit('plot-figure-result', JSON.parse(data))
	})

});

app.whenReady().then(() => {

    const window = new BrowserWindow (
		{
			width: 1200,
			height: 700,
			frame: true,
			title: 'MDF2D',
			webPreferences: {
				nodeIntegration: true
			}
		}
    )

    window.webContents.openDevTools()

    window.loadFile('./resources/pages/index.html')
  
})
  
app.on('window-all-closed', () => {

	if (process.platform !== 'darwin') {

		app.quit()

	}

})