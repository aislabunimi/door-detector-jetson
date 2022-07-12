model = Model()

ros = roslibpy.Ros(host='192.168.192.62', port=9090)
ros.run()

print(ros.is_connected)