from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RX, RY, RZ, X
from mindquantum import Rzz
import numpy as np


param_names = [
    'alphaRY0', 'betaRY0', 'gammaRY0', 'deltaRZZ0', 'alphaRY1', 'betaRY1', 'gammaRY1', 'deltaRZZ1',
    'alphaRY2', 'betaRY2', 'gammaRY2', 'deltaRZZ2', 'alphaRY3', 'betaRY3', 'gammaRY3', 'deltaRZZ3',
    'alphaRY4', 'betaRY4', 'gammaRY4', 'deltaRZZ4', 'alphaRY5', 'betaRY5', 'gammaRY5', 'deltaRZZ5',
    'alphaRY6', 'betaRY6', 'gammaRY6', 'deltaRZZ6', 'alphaRY7', 'betaRY7', 'gammaRY7', 'deltaRZZ7',
    'alphaRY8', 'betaRY8', 'gammaRY8', 'deltaRZZ8', 'alphaRY9', 'betaRY9', 'gammaRY9',
    'alphaRZ0', 'betaRZ0', 'gammaRZ0', 'alphaRZ1', 'betaRZ1', 'gammaRZ1',
    'alphaRZ2', 'betaRZ2', 'gammaRZ2', 'alphaRZ3', 'betaRZ3', 'gammaRZ3',
    'alphaRZ4', 'betaRZ4', 'gammaRZ4', 'alphaRZ5', 'betaRZ5', 'gammaRZ5',
    'alphaRZ6', 'betaRZ6', 'gammaRZ6', 'alphaRZ7', 'betaRZ7', 'gammaRZ7',
    'alphaRZ8', 'betaRZ8', 'gammaRZ8', 'alphaRZ9', 'betaRZ9', 'gammaRZ9',
    'alphaRZ10', 'betaRZ10', 'gammaRZ10',
    'alphaRY11', 'betaRY11', 'gammaRY11', 'deltaRZZ11', 'alphaRY12', 'betaRY12', 'gammaRY12', 'deltaRZZ12',
    'alphaRY13', 'betaRY13', 'gammaRY13', 'deltaRZZ13', 'alphaRY14', 'betaRY14', 'gammaRY14', 'deltaRZZ14',
    'alphaRY15', 'betaRY15', 'gammaRY15', 'deltaRZZ15', 'alphaRY16', 'betaRY16', 'gammaRY16', 'deltaRZZ16',
    'alphaRY17', 'betaRY17', 'gammaRY17', 'deltaRZZ17', 'alphaRY18', 'betaRY18', 'gammaRY18', 'deltaRZZ18',
    'alphaRY19', 'betaRY19', 'gammaRY19', 'deltaRZZ19', 'alphaRY20', 'betaRY20', 'gammaRY20',
    'alphaRZ11', 'betaRZ11', 'gammaRZ11', 'alphaRZ12', 'betaRZ12', 'gammaRZ12',
    'alphaRZ13', 'betaRZ13', 'gammaRZ13', 'alphaRZ14', 'betaRZ14', 'gammaRZ14',
    'alphaRZ15', 'betaRZ15', 'gammaRZ15', 'alphaRZ16', 'betaRZ16', 'gammaRZ16',
    'alphaRZ17', 'betaRZ17', 'gammaRZ17', 'alphaRZ18', 'betaRZ18', 'gammaRZ18',
    'alphaRZ19', 'betaRZ19', 'gammaRZ19', 'alphaRZ20', 'betaRZ20', 'gammaRZ20',
    'alphaRZ21', 'betaRZ21', 'gammaRZ21'
]


param_values =  [  -0.17759281396865845,    -0.1540789008140564,    -0.1607249230146408,    0.14787167310714722,    0.06327562034130096,    0.084840789437294,    0.07575220614671707,
    -0.4342263638973236,  -0.012835461646318436,    -0.012143362313508987,-0.003407673444598913, 0.23006671667099,    -0.19299013912677765,    -0.17771102488040924,    -0.1949072629213333,
    0.02551143802702427,  0.04699688032269478, 0.04545294865965843, 0.03872295096516609, 0.06465829908847809,  -0.0450802743434906,    -0.039081182330846786,    -0.054583922028541565,  -1.438895583152771,
    -0.03768976032733917,-0.06308242678642273,    -0.04812084138393402,    -0.3993844985961914,    -0.203721821308136,    -0.19235588610172272,    -0.20076555013656616,    -0.4789290130138397,    -0.07130955159664154, -0.0908961370587349,    -0.06478000432252884,    1.070915699005127,    0.8248322010040283,    0.8168172836303711,    0.8296489715576172,    -0.4414729177951813,    -0.447152704000473,
    -0.4230065643787384,    2.0959153175354004,    2.093622922897339,    2.0988411903381348,    -1.0396775007247925,    -1.0326112508773804,    -1.0249515771865845,    -0.31862449645996094,
    -0.29930204153060913,    -0.3217840790748596,    0.7868128418922424,    0.814002513885498,    0.779268205165863,    0.19191569089889526,    0.16338002681732178,    0.16277846693992615,
    -0.6958027482032776,    -0.6854299306869507,    -0.6973735094070435,    -0.40860477089881897,    -0.3894710838794708,    -0.398105651140213,    0.24426758289337158,    0.25063762068748474,
    0.2663727402687073,    0.15899719297885895,    0.1786818653345108,    0.14925657212734222,    0.012686226516962051,    -0.00603977357968688,    0.002698906697332859,    -0.04988237842917442,
    -0.04519437253475189,    -0.04624497890472412,    1.9584428071975708,    -0.05998636409640312,    -0.07488574087619781,    -0.05299944803118706,    -0.9817115664482117,    -0.18303999304771423,
    -0.1674189567565918,    -0.15876483917236328,    0.17074280977249146,    0.018465502187609673,    0.0022985772229731083,    0.01226864755153656,    -0.5011276602745056,    -0.05483485758304596,
    -0.05068044364452362,    -0.051356177777051926,    0.19202491641044617,    0.17739492654800415,    0.16858649253845215,    0.17134994268417358,    -0.44104334712028503,    -0.023958198726177216,
    -0.03292398899793625,    0.00019757216796278954,    -1.7645546197891235,    0.047173529863357544,    0.07462956756353378,    0.059911880642175674,    0.15659715235233307,    -0.2924749255180359,
    -0.28087443113327026,    -0.2850986123085022,    0.0861344039440155,    -0.48314520716667175,    -0.4979977011680603,    -0.48427531123161316,    -0.01866748370230198,
    0.0005197807913646102,    0.02127934992313385,    0.01964656263589859,    -0.009895693510770798,
    -0.019060632213950157,    -0.004094927571713924,    0.0035025402903556824,    0.003263697260990739,
    0.00016415676509495825,    0.004088351968675852,    -0.006375140510499477,    0.0006990788388065994,
    -0.028189565986394882,    -0.007161006797105074,    0.010675199329853058,    0.012796375900506973,
    -0.007329822052270174,    0.008453793823719025,    -0.009434955194592476,    -0.010683140717446804,
    0.03596973419189453,   0.01662869192659855,   -0.008719689212739468,  0.009879712015390396,   -0.018159715458750725,   -0.0024144260678440332,   0.010500708594918251,   -0.006211892701685429,
    0.011474652215838432,   0.0013101141666993499,    0.0056438748724758625,
    -0.006416656542569399  ]  

param_map = dict(zip(param_names, param_values))
encoder_num = 11

circuit = Circuit()

# ansatz 电路
for i in range(2):
    for j in range(encoder_num - 1):
        idx = encoder_num * i + j
        circuit += RY(param_map[f'alphaRY{idx}']).on(j)
        circuit += RY(param_map[f'betaRY{idx}']).on(j)
        circuit += RY(param_map[f'gammaRY{idx}']).on(j)
        if j != encoder_num - 2:
            circuit += Rzz(param_map[f'deltaRZZ{idx}']).on([j + 1, j + 2])

    for j in range(encoder_num):
        circuit += RZ(param_map[f'alphaRZ{encoder_num * i + j}']).on(j)
        circuit += RZ(param_map[f'betaRZ{encoder_num * i + j}']).on(j)
        circuit += RZ(param_map[f'gammaRZ{encoder_num * i + j}']).on(j)

    circuit += X.on(0, encoder_num - 1)
    for k in range(encoder_num - 1):
        if k != encoder_num - 2 and k != 0:
            circuit += X.on(k + 1, k)
            circuit += X.on(k - 1, k)

# 保存为 QASM
qasm_str = circuit.to_openqasm()
with open("qembedding1_ansatz.qasm", "w") as f:
    f.write(qasm_str)
print("QASM file saved to qembedding1_ansatz.qasm")
