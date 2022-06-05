import socket
import jsons
import argparse
import time

HOST = '127.0.0.1'
PORT = 8000
clientMessage = 'Hello!'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('-p', '--port', type=str, default='8080', help="port")
args = parser.parse_args()

if __name__ == '__main__':
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST, PORT))
    path = 'hash.txt'
    IPFS=[]

    with open(path) as f:
        IPFS = f.readlines()
    if args.port=="8080":
        LM_IPFS = IPFS[0].replace("\n", "")
    else :
        LM_IPFS = IPFS[1].replace("\n", "")

    GM_VerID = 1.0
    print("I'm "+ args.port)
    info_json={}

    # Local Model IPFS hash
    info_json['LM_IPFS'] = LM_IPFS
    print("info_json['LM_IPFS']", info_json['LM_IPFS'])

    # Global Model IPFS hash
    info_json['GM_VerID'] = GM_VerID
    print("info_json['GM_VerID']", info_json['GM_VerID'])

    # Client Number
    info_json['Client_Num'] = args.port
    jsonstr = jsons.dumps(info_json)
    print('jsonstr', jsonstr)

    client.sendall(jsonstr.encode())
        
    while True:
        Message = str(client.recv(1024), encoding='utf-8')
        if Message == None:
            Message="{""serverMessage"" : ""NOT YET"",""done_Num"": 0 }"
        print(Message)
        info_json = jsons.loads(Message)
        serverMessage = info_json['serverMessage']
        done_Num = info_json['done_Num']

        print('Server:', done_Num)
        print('Server:', serverMessage)
        first=0
        if done_Num ==2:
            break

    client.close()