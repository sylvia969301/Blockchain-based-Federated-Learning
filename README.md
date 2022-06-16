### Description :
- This project used Flower, Go Ethereum, Web3, IPFS, Socket, Digital signature to implement <b>Blockchain-based federated learning</b>
---
- 執行順序：
    - Run Go Ethereum Boot Node
    `$ bootnode --verbosity 9 --nodekey=boot.key`
    - Run Miner Node, set up network id they’re connect with
        1. Miner 1: <br>
        `$ geth --mine --miner.threads 2 --syncmode "full" --miner.gasprice=0 --networkid 8787 --bootnodes enode://bootnode -writeaddress --nodekey=boot.key@127.0.0.1:30301 --datadir miner1/data --unlock "0xca38618860d047b63592cab7CB26c6DF2554141F" --password password.txt --http --http.addr 0.0.0.0 --http.port 8547 --http.corsdomain '' --http.api 'admin,eth,net,web3,personal,miner' --ws --ws.addr 0.0.0.0 --ws.port 8548 --port 30304 --ws.origins '' --ws.api 'eth,net,web3' --nousb --allow-insecure-unlock`
        2. Miner 2: <br>
        `$ geth --mine --miner.threads 2 --syncmode "full" --miner.gasprice=0 --networkid 8787 --bootnodes enode://bootnode -writeaddress --nodekey=boot.key@127.0.0.1:30301 --datadir miner2/data --unlock "0x086e0Fcab268b9eE0C781a58D93D8de524D58fBf" --password password.txt --http --http.addr 0.0.0.0 --http.port 8549 --http.corsdomain '' --http.api 'admin,eth,net,web3,personal,miner' --ws --ws.addr 0.0.0.0 --ws.port 8549 --port 30305 --ws.origins '' --ws.api 'eth,net,web3' --nousb --allow-insecure-unlock`

    - Run IPFS daemon <br>
    `$ cd BCFL/ & ipfs daemon`
    - Flower server-client local training
        1. `$ cd BCFL/`
        2. `$ python server_8080.py`
        3. `$ python client_8080.py`
        4. `$ python server_8787.py`
        5. `$ python client_8787.py`
    - Run Task owner (socket_server)
        1. `$ python socket_server.py --deploy_contract True`
    - Client(s) pass their trained local model weight (IPFS hash) to task owner for validation & aggregation
        1. `$ python socket_client.py -p="8787"`
        2. `$ python socket_client.py -p="8080"`
        
 - Result : Task owner會回傳新Global Model Weight IPFS Hash, update `latest_GMhash.txt`  and clear `hash.txt`
