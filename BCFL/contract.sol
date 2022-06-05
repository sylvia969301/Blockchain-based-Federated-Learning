
contract StoreVar {

    string public ownerSig;
    string public modelWeightIPFSHash;

    event UpdateEvent(string _ownerSig,string modelWeightIPFSHash);
    event GetInfo(string _ownerSig, string modelWeightIPFSHash);

    function setFLUpdate(string memory _ownerSig,
                        string memory _weightIPFS) public{
        ownerSig = _ownerSig;
        modelWeightIPFSHash = _weightIPFS;

    }

    function getFLUpdate() public view returns (string memory, string memory) {
       return (ownerSig, modelWeightIPFSHash);
    }

}