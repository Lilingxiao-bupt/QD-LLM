from mindnlp.transforms import BasicTokenizer
from mindquantum.framework import MQLayer, MQN2Layer
from mindquantum.core.gates import RX, RY, RZ, X, H, CNOT
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum import Rzz
from mindquantum.simulator import Simulator
import mindspore.nn as nn
import mindspore.numpy as mnp
import numpy as np
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore.common.initializer import Uniform, HeUniform
import mindspore.common.initializer as init
import mindspore as ms
import torch
import torch.nn as tnn
import mindspore.ops as P
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel
import json
from datasets import load_dataset, DownloadConfig


def GenerateEmbeddingHamiltonian(dims, n_qubits):
    hams = []
    for i in range(dims):
        s = ''
        for j, k in enumerate(bin(i + 1)[-1:1:-1]):
            if k == '1':
                s = s + 'Z' + str(j) + ' '
        hams.append(Hamiltonian(QubitOperator(s)))
    return hams


def GenerateEncoderCircuit(n_qubits, prefix=''):
    if prefix and prefix[-1] != '_':
        prefix += '_'
    circ = Circuit()
    for i in range(n_qubits):
        circ += RX(prefix + str(i)).on(i) 
    return circ.as_encoder()


def GenerateAnsatzCircuit(n_qubits, layers, prefix=''):
    if prefix and prefix[-1] != '_':
        prefix += '_'
    circ = Circuit()
    for l in range(layers):
        for i in range(n_qubits):
            circ += RY(prefix + str(l) + '_' + str(i)).on(i)  
        for i in range(l % 2, n_qubits, 2):
            if i < n_qubits and i + 1 < n_qubits:
                circ += X.on(i + 1, i)
    return circ.as_ansatz()


# def QEmbedding(num_embedding, embedding_dim, text_len, n_class, n_threads=32):
def QEmbedding1(encoder_num, n_class, withlabel, n_threads=32):
   
    if withlabel == 'yes':
        if n_class == 2:
            ansatz_num = encoder_num + n_class
            encoder_iris_qmcc = Circuit()
            for i in range(encoder_num):
                encoder_iris_qmcc += H.on(i)
            for i in range(encoder_num):
                encoder_iris_qmcc += RX(f'alpha{i}').on(i)
                encoder_iris_qmcc += RY(f'alpha{i}').on(i)
                encoder_iris_qmcc += RZ(f'alpha{i}').on(i)
            for i in range(encoder_num):
                if i != encoder_num - 1:
                    encoder_iris_qmcc += X.on(i + 1, i)
                    encoder_iris_qmcc += X.on(0, encoder_num - 1)
            # encoder_iris_qmcc = encoder_iris_qmcc.no_grad()  

            ansatz_iris_qmcc = Circuit()
            for i in range(ansatz_num):
                for j in range(encoder_num):
                    ansatz_iris_qmcc += RY(f'beta{j + ansatz_num * i}').on(j)  

                for k in range(encoder_num - 2):
                    if k == 0:
                        ansatz_iris_qmcc += RX(f'beta{(0) + ansatz_num * i}').on(0)
                        ansatz_iris_qmcc += X.on(1, 0)  
                        ansatz_iris_qmcc += X.on(2, 0)  
                    if k != 0:
                        ansatz_iris_qmcc += RX(f'beta{(k) + ansatz_num * i}').on(k)
                        ansatz_iris_qmcc += X.on(k + 1, k)
                        ansatz_iris_qmcc += X.on(k + 2, k)
                    elif k == encoder_num - 2:
                        ansatz_iris_qmcc += RX(f'beta{(k) + ansatz_num * i}').on(k)
                        ansatz_iris_qmcc += X.on(k + 1, k)
                        ansatz_iris_qmcc += X.on(0, k)
                    elif k == encoder_num - 1:
                        ansatz_iris_qmcc += RX(f'beta{(k) + ansatz_num * i}').on(k)
                        ansatz_iris_qmcc += X.on(0, k)
                        ansatz_iris_qmcc += X.on(1, k)

                ansatz_iris_qmcc += RZ(f'beta{(ansatz_num - 2) + ansatz_num * i}').on(ansatz_num - 2)
                ansatz_iris_qmcc += RY(f'beta{(ansatz_num - 1) + ansatz_num * i}').on(ansatz_num - 1)

            circuit_iris_qmcc = encoder_iris_qmcc.as_encoder() + ansatz_iris_qmcc.as_ansatz()
            # print(circuit_iris_qmcc)
            hams_qmcc = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [ansatz_num - 2, ansatz_num - 1]]

        elif n_class == 3:
            ansatz_num = encoder_num + n_class
            encoder_iris_qmcc = Circuit()
            for i in range(encoder_num):
                encoder_iris_qmcc += H.on(i)
            for i in range(encoder_num):
                encoder_iris_qmcc += RX(f'alpha{i}').on(i)
                encoder_iris_qmcc += RY(f'alpha{i}').on(i)
                encoder_iris_qmcc += RZ(f'alpha{i}').on(i)
            for i in range(encoder_num):
                if i != encoder_num - 1:
                    encoder_iris_qmcc += X.on(i + 1, i)
                    encoder_iris_qmcc += X.on(0, encoder_num - 1)
            # encoder_iris_qmcc = encoder_iris_qmcc.no_grad()  

            ansatz_iris_qmcc = Circuit()
            for i in range(ansatz_num):
                for j in range(encoder_num):
                    ansatz_iris_qmcc += RY(f'beta{j + ansatz_num * i}').on(j)  

                for k in range(encoder_num - 2):
                    if k == 0:
                        ansatz_iris_qmcc += RX(f'beta{(0) + ansatz_num * i}').on(0)
                        ansatz_iris_qmcc += X.on(1, 0)
                        ansatz_iris_qmcc += X.on(2, 0)
                    if k != 0:
                        ansatz_iris_qmcc += RX(f'beta{(k) + ansatz_num * i}').on(k)
                        ansatz_iris_qmcc += X.on(k + 1, k)
                        ansatz_iris_qmcc += X.on(k + 2, k)
                    elif k == encoder_num - 2:
                        ansatz_iris_qmcc += RX(f'beta{(k) + ansatz_num * i}').on(k)
                        ansatz_iris_qmcc += X.on(k + 1, k)
                        ansatz_iris_qmcc += X.on(0, k)
                    elif k == encoder_num - 1:
                        ansatz_iris_qmcc += RX(f'beta{(k) + ansatz_num * i}').on(k)
                        ansatz_iris_qmcc += X.on(0, k)
                        ansatz_iris_qmcc += X.on(1, k)

                ansatz_iris_qmcc += RZ(f'beta{(ansatz_num-3) + ansatz_num * i}').on(ansatz_num-3)
                ansatz_iris_qmcc += RZ(f'beta{(ansatz_num-2) + ansatz_num * i}').on(ansatz_num-2)
                ansatz_iris_qmcc += RY(f'beta{(ansatz_num-1) + ansatz_num * i}').on(ansatz_num-1)

            circuit_iris_qmcc = encoder_iris_qmcc.as_encoder() + ansatz_iris_qmcc.as_ansatz()
            # print(circuit_iris_qmcc)
            hams_qmcc = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [ansatz_num - 3, ansatz_num - 2, ansatz_num - 1]]

        elif n_class == 4:
            ansatz_num = encoder_num + n_class
            encoder_iris_qmcc = Circuit()
            for i in range(encoder_num):
                encoder_iris_qmcc += H.on(i)
            for i in range(encoder_num):
                encoder_iris_qmcc += RX(f'alpha{i}').on(i)
                encoder_iris_qmcc += RY(f'alpha{i}').on(i)
                encoder_iris_qmcc += RZ(f'alpha{i}').on(i)
            for i in range(encoder_num):
                if i != encoder_num - 1:
                    encoder_iris_qmcc += X.on(i + 1, i)
                    encoder_iris_qmcc += X.on(0, encoder_num - 1)
            # encoder_iris_qmcc = encoder_iris_qmcc.no_grad()   

            ansatz_iris_qmcc = Circuit()
            for i in range(ansatz_num):
                for j in range(encoder_num):
                    ansatz_iris_qmcc += RY(f'beta{j + ansatz_num * i}').on(j)   

                for k in range(encoder_num - 2):
                    if k == 0:
                        ansatz_iris_qmcc += RX(f'beta{(0) + ansatz_num * i}').on(0)
                        ansatz_iris_qmcc += X.on(1, 0)  
                        ansatz_iris_qmcc += X.on(2, 0)  
                    if k != 0:
                        ansatz_iris_qmcc += RX(f'beta{(k) + ansatz_num * i}').on(k)
                        ansatz_iris_qmcc += X.on(k + 1, k)
                        ansatz_iris_qmcc += X.on(k + 2, k)
                    elif k == encoder_num - 2:
                        ansatz_iris_qmcc += RX(f'beta{(k) + ansatz_num * i}').on(k)
                        ansatz_iris_qmcc += X.on(k + 1, k)
                        ansatz_iris_qmcc += X.on(0, k)
                    elif k == encoder_num - 1:
                        ansatz_iris_qmcc += RX(f'beta{(k) + ansatz_num * i}').on(k)
                        ansatz_iris_qmcc += X.on(0, k)
                        ansatz_iris_qmcc += X.on(1, k)

                ansatz_iris_qmcc += RZ(f'beta{(ansatz_num-4) + ansatz_num * i}').on(ansatz_num-4)
                ansatz_iris_qmcc += RZ(f'beta{(ansatz_num-3) + ansatz_num * i}').on(ansatz_num-3)
                ansatz_iris_qmcc += RZ(f'beta{(ansatz_num-2) + ansatz_num * i}').on(ansatz_num-2)
                ansatz_iris_qmcc += RY(f'beta{(ansatz_num-1) + ansatz_num * i}').on(ansatz_num-1)

            circuit_iris_qmcc = encoder_iris_qmcc.as_encoder() + ansatz_iris_qmcc.as_ansatz()
            # print(circuit_iris_qmcc)
            hams_qmcc = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [ansatz_num - 4, ansatz_num - 3, ansatz_num - 2, ansatz_num - 1]]

    elif withlabel == 'no':
        # ansatz_num = encoder_num
        # encoder_iris_qmcc = Circuit()
        # for i in range(encoder_num):
        #     encoder_iris_qmcc += H.on(i)
        # for i in range(encoder_num):
        #     encoder_iris_qmcc += RY(f'alpha{i}').on(i)
        #     encoder_iris_qmcc += RX(f'alpha{i}').on(i)
        # # encoder_iris_qmcc = encoder_iris_qmcc.no_grad()  
        #
        # ansatz_iris_qmcc = Circuit()
        # for i in range(ansatz_num):
        #     for j in range(encoder_num):
        #         ansatz_iris_qmcc += RY(f'alphaRY{j + ansatz_num * i}').on(j)  
        #     ansatz_iris_qmcc += X.on(0, encoder_num - 1) 
        #     ansatz_iris_qmcc += X.on(1, encoder_num - 1)  
        #     for k in range(encoder_num - 1):
        #         if k != encoder_num - 2:
        #             ansatz_iris_qmcc += X.on(k + 1, k)
        #             ansatz_iris_qmcc += X.on(k + 2, k)
        #
        #     for j in range(encoder_num):
        #         ansatz_iris_qmcc += RX(f'alphaRX{j + ansatz_num * i}').on(j)  
        #     ansatz_iris_qmcc += X.on(0, encoder_num - 1)  
        #     ansatz_iris_qmcc += X.on(1, encoder_num - 1)  
        #     for k in range(encoder_num - 1):
        #         if k != encoder_num - 2:
        #             ansatz_iris_qmcc += X.on(k + 1, k)
        #             ansatz_iris_qmcc += X.on(k + 2, k)
        #
        #     for j in range(encoder_num):
        #         ansatz_iris_qmcc += RZ(f'alphaRZ{j + ansatz_num * i}').on(j)  
        #     ansatz_iris_qmcc += X.on(0, encoder_num - 1)  
        #     ansatz_iris_qmcc += X.on(1, encoder_num - 1) 
        #     for k in range(encoder_num - 1):
        #         if k != encoder_num - 2:
        #             ansatz_iris_qmcc += X.on(k + 1, k)
        #             ansatz_iris_qmcc += X.on(k + 2, k)
        #
        #
        #     for j in range(encoder_num):
        #         ansatz_iris_qmcc += RY(f'betaRY{j + ansatz_num * i}').on(j)  
        #     ansatz_iris_qmcc += X.on(encoder_num - 1, 0)  
        #     ansatz_iris_qmcc += X.on(encoder_num - 2, 0)  
        #     ansatz_iris_qmcc += X.on(0, 1)  
        #     ansatz_iris_qmcc += X.on(encoder_num - 1, 1) 
        #     for k in range(encoder_num - 1):
        #         if k > 1:
        #             ansatz_iris_qmcc += X.on(k - 1, k)
        #             ansatz_iris_qmcc += X.on(k - 2, k)
        #
        #     for j in range(encoder_num):
        #         ansatz_iris_qmcc += RX(f'betaRX{j + ansatz_num * i}').on(j)  
        #     ansatz_iris_qmcc += X.on(encoder_num - 1, 0) 
        #     ansatz_iris_qmcc += X.on(encoder_num - 2, 0) 
        #     ansatz_iris_qmcc += X.on(0, 1)  
        #     ansatz_iris_qmcc += X.on(encoder_num - 1, 1)  
        #     for k in range(encoder_num - 1):
        #         if k > 1:
        #             ansatz_iris_qmcc += X.on(k - 1, k)
        #             ansatz_iris_qmcc += X.on(k - 2, k)
        #
        #     for j in range(encoder_num):
        #         ansatz_iris_qmcc += RZ(f'betaRZ{j + ansatz_num * i}').on(j)  
        #     ansatz_iris_qmcc += X.on(encoder_num - 1, 0)  
        #     ansatz_iris_qmcc += X.on(encoder_num - 2, 0)  
        #     ansatz_iris_qmcc += X.on(0, 1)  
        #     ansatz_iris_qmcc += X.on(encoder_num - 1, 1)  
        #     for k in range(encoder_num - 1):
        #         if k > 1:
        #             ansatz_iris_qmcc += X.on(k - 1, k)
        #             ansatz_iris_qmcc += X.on(k - 2, k)
        #
        #     for j in range(encoder_num):
        #         ansatz_iris_qmcc += RY(f'gammaRY{j + ansatz_num * i}').on(j)  
        #     ansatz_iris_qmcc += X.on(0, encoder_num - 1)  
        #     for k in range(encoder_num - 1):
        #         if k != encoder_num - 2:
        #             ansatz_iris_qmcc += X.on(k + 1, k)
        #
        #     for j in range(encoder_num):
        #         ansatz_iris_qmcc += RX(f'gammaRX{j + ansatz_num * i}').on(j)  
        #     ansatz_iris_qmcc += X.on(0, encoder_num - 1)  
        #     for k in range(encoder_num - 1):
        #         if k != encoder_num - 2:
        #             ansatz_iris_qmcc += X.on(k + 1, k)
        #
        #     for j in range(encoder_num):
        #         ansatz_iris_qmcc += RZ(f'gammaRZ{j + ansatz_num * i}').on(j)  
        #     ansatz_iris_qmcc += X.on(0, encoder_num - 1)  
        #     for k in range(encoder_num - 1):
        #         if k != encoder_num - 2:
        #             ansatz_iris_qmcc += X.on(k + 1, k)

        ansatz_num = encoder_num
        encoder_iris_qmcc = Circuit()
        for i in range(encoder_num):
            encoder_iris_qmcc += RX(f'theta{i}').on(i)
        # encoder_iris_qmcc = encoder_iris_qmcc.no_grad()  

        ansatz_iris_qmcc = Circuit()
        for i in range(2):
            for j in range(encoder_num-1):
                ansatz_iris_qmcc += RY(f'alphaRY{j + ansatz_num * i}').on(j)  
                ansatz_iris_qmcc += RY(f'betaRY{j + ansatz_num * i}').on(j)
                ansatz_iris_qmcc += RY(f'gammaRY{j + ansatz_num * i}').on(j)
                if j != encoder_num - 2:
                    ansatz_iris_qmcc += Rzz(f'deltaRZZ{j + ansatz_num * i}').on([j + 1, j + 2])
                    # ansatz_iris_qmcc += X.on(j + 1, j)
                    # ansatz_iris_qmcc += X.on(j + 2, j)

            for j in range(encoder_num):
                ansatz_iris_qmcc += RZ(f'alphaRZ{j + ansatz_num * i}').on(j)   
                ansatz_iris_qmcc += RZ(f'betaRZ{j + ansatz_num * i}').on(j)  
                ansatz_iris_qmcc += RZ(f'gammaRZ{j + ansatz_num * i}').on(j) 
            ansatz_iris_qmcc += X.on(0, encoder_num - 1)  
            for k in range(encoder_num - 1):
                if k != encoder_num - 2 and k != 0 :
                    ansatz_iris_qmcc += X.on(k + 1, k)
                    ansatz_iris_qmcc += X.on(k - 1, k)

        circuit_iris_qmcc = encoder_iris_qmcc.as_encoder() + ansatz_iris_qmcc.as_ansatz()
        # print(circuit_iris_qmcc)
        hams_qmcc = [Hamiltonian(QubitOperator(f'Z{i}')) for i in range(ansatz_num)]

        # ansatz_num = encoder_num
        # encoder_iris_qmcc = Circuit()
        # for i in range(encoder_num):
        #     encoder_iris_qmcc += H.on(i)
        # for i in range(encoder_num):
        #     encoder_iris_qmcc += RX(f'alpha{i}').on(i)
        #     encoder_iris_qmcc += RY(f'alpha{i}').on(i)
        #     # encoder_iris_qmcc += RZ(f'alpha{i}').on(i)
        # for i in range(encoder_num):
        #     if i == 0:
        #         encoder_iris_qmcc += X.on(encoder_num-1, 0)
        #         encoder_iris_qmcc += X.on(1, 0)
        #     elif i != encoder_num-1 and i != 0:
        #         encoder_iris_qmcc += X.on(i - 1, i)
        #         encoder_iris_qmcc += X.on(i + 1, i)
        #     elif i == encoder_num-1:
        #         encoder_iris_qmcc += X.on(encoder_num-2, encoder_num-1)
        #         encoder_iris_qmcc += X.on(0, encoder_num-1)
        # encoder_iris_qmcc = encoder_iris_qmcc.no_grad()
        #
        # ansatz_iris_qmcc = Circuit()
        # for i in range(ansatz_num):
        #     for j in range(encoder_num):
        #         ansatz_iris_qmcc += RX(f'betaRX{j + ansatz_num * i}').on(j)
        #         ansatz_iris_qmcc += RY(f'betaRY{j + ansatz_num * i}').on(j)
        #
        #         # ansatz_iris_qmcc += RX(f'gamma{j + ansatz_num * i}').on(j)
        #         # ansatz_iris_qmcc += RY(f'gamma{j + ansatz_num * i}').on(j)
        #
        #     for k in range(encoder_num - 2):
        #         if k == 0:
        #             ansatz_iris_qmcc += X.on(1, 0) 
        #             ansatz_iris_qmcc += X.on(2, 0)  
        #         if k != 0:
        #             ansatz_iris_qmcc += RX(f'betaRX{(k) + ansatz_num * i}').on(k)
        #             ansatz_iris_qmcc += RY(f'betaRY{(k) + ansatz_num * i}').on(k)
        #             ansatz_iris_qmcc += X.on(k+1, k)
        #             ansatz_iris_qmcc += X.on(k+2, k)
        #             # ansatz_iris_qmcc += RY(f'phiRY{(k) + ansatz_num * i}').on(k)
        #             # ansatz_iris_qmcc += X.on(k + 1, k)
        #             # ansatz_iris_qmcc += X.on(k + 2, k)
        #         elif k == encoder_num - 2:
        #             ansatz_iris_qmcc += RX(f'betaRX{(k) + ansatz_num * i}').on(k)
        #             ansatz_iris_qmcc += RY(f'betaRY{(k) + ansatz_num * i}').on(k)
        #             ansatz_iris_qmcc += X.on(k + 1, k)
        #             ansatz_iris_qmcc += X.on(0, k)
        #             # ansatz_iris_qmcc += RY(f'phiRY{(k) + ansatz_num * i}').on(k)
        #             # ansatz_iris_qmcc += X.on(k + 1, k)
        #             # ansatz_iris_qmcc += X.on(0, k)
        #         elif k == encoder_num - 1:
        #             ansatz_iris_qmcc += RX(f'betaRX{(k) + ansatz_num * i}').on(k)
        #             ansatz_iris_qmcc += RY(f'betaRY{(k) + ansatz_num * i}').on(k)
        #             ansatz_iris_qmcc += X.on(0, k)
        #             ansatz_iris_qmcc += X.on(1, k)
        #             # ansatz_iris_qmcc += RY(f'phiRY{(k) + ansatz_num * i}').on(k)
        #             # ansatz_iris_qmcc += X.on(0, k)
        #             # ansatz_iris_qmcc += X.on(1, k)
        #         ansatz_iris_qmcc += RZ(f'betaRZ{(k) + ansatz_num * i}').on(k)
        #         ansatz_iris_qmcc += RY(f'betaRY{(k) + ansatz_num * i}').on(k)
        #         # ansatz_iris_qmcc += RZ(f'phiRZ{(k) + ansatz_num * i}').on(k)
        #         # ansatz_iris_qmcc += RY(f'phiRY{(k) + ansatz_num * i}').on(k)
        #
        #     ansatz_iris_qmcc += RY(f'betaRY{i}').on(k)
        #     ansatz_iris_qmcc += RY(f'thetaRY{i}').on(k)
        # circuit_iris_qmcc = encoder_iris_qmcc.as_encoder() + ansatz_iris_qmcc.as_ansatz()
        # # print(circuit_iris_qmcc)
        # hams_qmcc = [Hamiltonian(QubitOperator(f'Z{i}')) for i in range(encoder_num)]

    grad_ops = Simulator('mqvector', circuit_iris_qmcc.n_qubits).get_expectation_with_grad(hams_qmcc, circuit_iris_qmcc, parallel_worker=n_threads)
    MQ = MQLayer(grad_ops)
    return MQ


def QEmbedding2(encoder_num, n_class, withlabel, n_threads=32):
    
    if withlabel == 'yes':
        if n_class == 2:
            ansatz_num = encoder_num + n_class
            encoder_iris_qmcc = Circuit()
            for i in range(encoder_num):
                encoder_iris_qmcc += H.on(i)
            for i in range(encoder_num):
                encoder_iris_qmcc += RX(f'alpha{i}').on(i)
                encoder_iris_qmcc += RY(f'alpha{i}').on(i)
                encoder_iris_qmcc += RZ(f'alpha{i}').on(i)
            for i in range(encoder_num):
                if i != encoder_num - 1:
                    encoder_iris_qmcc += X.on(i + 1, i)
                    encoder_iris_qmcc += X.on(0, encoder_num - 1)
            # encoder_iris_qmcc = encoder_iris_qmcc.no_grad()  

            ansatz_iris_qmcc = Circuit()
            for i in range(ansatz_num):
                for j in range(encoder_num):
                    ansatz_iris_qmcc += RY(f'beta{j + ansatz_num * i}').on(j)  

                for k in range(encoder_num - 2):
                    if k == 0:
                        ansatz_iris_qmcc += RX(f'beta{(0) + ansatz_num * i}').on(0)
                        ansatz_iris_qmcc += X.on(1, 0) 
                        ansatz_iris_qmcc += X.on(2, 0)  
                    if k != 0:
                        ansatz_iris_qmcc += RX(f'beta{(k) + ansatz_num * i}').on(k)
                        ansatz_iris_qmcc += X.on(k + 1, k)
                        ansatz_iris_qmcc += X.on(k + 2, k)
                    elif k == encoder_num - 2:
                        ansatz_iris_qmcc += RX(f'beta{(k) + ansatz_num * i}').on(k)
                        ansatz_iris_qmcc += X.on(k + 1, k)
                        ansatz_iris_qmcc += X.on(0, k)
                    elif k == encoder_num - 1:
                        ansatz_iris_qmcc += RX(f'beta{(k) + ansatz_num * i}').on(k)
                        ansatz_iris_qmcc += X.on(0, k)
                        ansatz_iris_qmcc += X.on(1, k)

                ansatz_iris_qmcc += RZ(f'beta{(ansatz_num - 2) + ansatz_num * i}').on(ansatz_num - 2)
                ansatz_iris_qmcc += RY(f'beta{(ansatz_num - 1) + ansatz_num * i}').on(ansatz_num - 1)

            circuit_iris_qmcc = encoder_iris_qmcc.as_encoder() + ansatz_iris_qmcc.as_ansatz()
            # print(circuit_iris_qmcc)
            hams_qmcc = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [ansatz_num - 2, ansatz_num - 1]]

        elif n_class == 3:
            ansatz_num = encoder_num + n_class
            encoder_iris_qmcc = Circuit()
            for i in range(encoder_num):
                encoder_iris_qmcc += H.on(i)
            for i in range(encoder_num):
                encoder_iris_qmcc += RX(f'alpha{i}').on(i)
                encoder_iris_qmcc += RY(f'alpha{i}').on(i)
                encoder_iris_qmcc += RZ(f'alpha{i}').on(i)
            for i in range(encoder_num):
                if i != encoder_num - 1:
                    encoder_iris_qmcc += X.on(i + 1, i)
                    encoder_iris_qmcc += X.on(0, encoder_num - 1)
            # encoder_iris_qmcc = encoder_iris_qmcc.no_grad()  

            ansatz_iris_qmcc = Circuit()
            for i in range(ansatz_num):
                for j in range(encoder_num):
                    ansatz_iris_qmcc += RY(f'beta{j + ansatz_num * i}').on(j)  

                for k in range(encoder_num - 2):
                    if k == 0:
                        ansatz_iris_qmcc += RX(f'beta{(0) + ansatz_num * i}').on(0)
                        ansatz_iris_qmcc += X.on(1, 0)
                        ansatz_iris_qmcc += X.on(2, 0)
                    if k != 0:
                        ansatz_iris_qmcc += RX(f'beta{(k) + ansatz_num * i}').on(k)
                        ansatz_iris_qmcc += X.on(k + 1, k)
                        ansatz_iris_qmcc += X.on(k + 2, k)
                    elif k == encoder_num - 2:
                        ansatz_iris_qmcc += RX(f'beta{(k) + ansatz_num * i}').on(k)
                        ansatz_iris_qmcc += X.on(k + 1, k)
                        ansatz_iris_qmcc += X.on(0, k)
                    elif k == encoder_num - 1:
                        ansatz_iris_qmcc += RX(f'beta{(k) + ansatz_num * i}').on(k)
                        ansatz_iris_qmcc += X.on(0, k)
                        ansatz_iris_qmcc += X.on(1, k)

                ansatz_iris_qmcc += RZ(f'beta{(ansatz_num-3) + ansatz_num * i}').on(ansatz_num-3)
                ansatz_iris_qmcc += RZ(f'beta{(ansatz_num-2) + ansatz_num * i}').on(ansatz_num-2)
                ansatz_iris_qmcc += RY(f'beta{(ansatz_num-1) + ansatz_num * i}').on(ansatz_num-1)

            circuit_iris_qmcc = encoder_iris_qmcc.as_encoder() + ansatz_iris_qmcc.as_ansatz()
            # print(circuit_iris_qmcc)
            hams_qmcc = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [ansatz_num - 3, ansatz_num - 2, ansatz_num - 1]]

        elif n_class == 4:
            ansatz_num = encoder_num + n_class
            encoder_iris_qmcc = Circuit()
            for i in range(encoder_num):
                encoder_iris_qmcc += RX(f'alpha{i}').on(i)
                encoder_iris_qmcc += RY(f'alpha{i}').on(i)
                encoder_iris_qmcc += RZ(f'alpha{i}').on(i)
            for i in range(encoder_num):
                if i != encoder_num - 1:
                    encoder_iris_qmcc += X.on(i + 1, i)
                    encoder_iris_qmcc += X.on(0, encoder_num - 1)
            # encoder_iris_qmcc = encoder_iris_qmcc.no_grad()   

            ansatz_iris_qmcc = Circuit()
            for i in range(ansatz_num):
                for j in range(encoder_num):
                    ansatz_iris_qmcc += RY(f'beta{j + ansatz_num * i}').on(j) 

                for k in range(encoder_num - 2):
                    if k == 0:
                        ansatz_iris_qmcc += RX(f'beta{(0) + ansatz_num * i}').on(0)
                        ansatz_iris_qmcc += X.on(1, 0)  
                        ansatz_iris_qmcc += X.on(2, 0)  
                    if k != 0:
                        ansatz_iris_qmcc += RX(f'beta{(k) + ansatz_num * i}').on(k)
                        ansatz_iris_qmcc += X.on(k + 1, k)
                        ansatz_iris_qmcc += X.on(k + 2, k)
                    elif k == encoder_num - 2:
                        ansatz_iris_qmcc += RX(f'beta{(k) + ansatz_num * i}').on(k)
                        ansatz_iris_qmcc += X.on(k + 1, k)
                        ansatz_iris_qmcc += X.on(0, k)
                    elif k == encoder_num - 1:
                        ansatz_iris_qmcc += RX(f'beta{(k) + ansatz_num * i}').on(k)
                        ansatz_iris_qmcc += X.on(0, k)
                        ansatz_iris_qmcc += X.on(1, k)

                ansatz_iris_qmcc += RZ(f'beta{(ansatz_num-4) + ansatz_num * i}').on(ansatz_num-4)
                ansatz_iris_qmcc += RZ(f'beta{(ansatz_num-3) + ansatz_num * i}').on(ansatz_num-3)
                ansatz_iris_qmcc += RZ(f'beta{(ansatz_num-2) + ansatz_num * i}').on(ansatz_num-2)
                ansatz_iris_qmcc += RY(f'beta{(ansatz_num-1) + ansatz_num * i}').on(ansatz_num-1)

            circuit_iris_qmcc = encoder_iris_qmcc.as_encoder() + ansatz_iris_qmcc.as_ansatz()
            # print(circuit_iris_qmcc)
            hams_qmcc = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [ansatz_num - 4, ansatz_num - 3, ansatz_num - 2, ansatz_num - 1]]

    elif withlabel == 'no':
        ansatz_num = encoder_num
        encoder_iris_qmcc = Circuit()
        for i in range(encoder_num):
            encoder_iris_qmcc += H.on(i)
        for i in range(encoder_num):
            encoder_iris_qmcc += RY(f'alpha{i}').on(i)
            encoder_iris_qmcc += RX(f'alpha{i}').on(i)
        # encoder_iris_qmcc = encoder_iris_qmcc.no_grad()   

        ansatz_iris_qmcc = Circuit()
        for i in range(ansatz_num):
            for j in range(encoder_num):
                ansatz_iris_qmcc += RY(f'alphaRY{j + ansatz_num * i}').on(j)  
            ansatz_iris_qmcc += X.on(0, encoder_num - 1)  
            ansatz_iris_qmcc += X.on(1, encoder_num - 1)  
            for k in range(encoder_num - 1):
                if k != encoder_num - 2:
                    ansatz_iris_qmcc += X.on(k + 1, k)
                    ansatz_iris_qmcc += X.on(k + 2, k)

            for j in range(encoder_num):
                ansatz_iris_qmcc += RX(f'alphaRX{j + ansatz_num * i}').on(j)  
            ansatz_iris_qmcc += X.on(0, encoder_num - 1)  
            ansatz_iris_qmcc += X.on(1, encoder_num - 1)  
            for k in range(encoder_num - 1):
                if k != encoder_num - 2:
                    ansatz_iris_qmcc += X.on(k + 1, k)
                    ansatz_iris_qmcc += X.on(k + 2, k)

            for j in range(encoder_num):
                ansatz_iris_qmcc += RZ(f'alphaRZ{j + ansatz_num * i}').on(j)  
            ansatz_iris_qmcc += X.on(0, encoder_num - 1)  
            ansatz_iris_qmcc += X.on(1, encoder_num - 1)  
            for k in range(encoder_num - 1):
                if k != encoder_num - 2:
                    ansatz_iris_qmcc += X.on(k + 1, k)
                    ansatz_iris_qmcc += X.on(k + 2, k)

            for j in range(encoder_num):
                ansatz_iris_qmcc += RY(f'betaRY{j + ansatz_num * i}').on(j)  
            ansatz_iris_qmcc += X.on(0, encoder_num - 1)  
            ansatz_iris_qmcc += X.on(1, encoder_num - 1)  
            for k in range(encoder_num - 1):
                if k != encoder_num - 2:
                    ansatz_iris_qmcc += X.on(k + 1, k)
                    ansatz_iris_qmcc += X.on(k + 2, k)

            for j in range(encoder_num):
                ansatz_iris_qmcc += RX(f'betaRX{j + ansatz_num * i}').on(j)  
            ansatz_iris_qmcc += X.on(0, encoder_num - 1)  
            ansatz_iris_qmcc += X.on(1, encoder_num - 1)  
            for k in range(encoder_num - 1):
                if k != encoder_num - 2:
                    ansatz_iris_qmcc += X.on(k + 1, k)
                    ansatz_iris_qmcc += X.on(k + 2, k)

            for j in range(encoder_num):
                ansatz_iris_qmcc += RZ(f'betaRZ{j + ansatz_num * i}').on(j)  
            ansatz_iris_qmcc += X.on(0, encoder_num - 1)  
            ansatz_iris_qmcc += X.on(1, encoder_num - 1)  
            for k in range(encoder_num - 1):
                if k != encoder_num - 2:
                    ansatz_iris_qmcc += X.on(k + 1, k)
                    ansatz_iris_qmcc += X.on(k + 2, k)

            for j in range(encoder_num):
                ansatz_iris_qmcc += RY(f'gammaRY{j + ansatz_num * i}').on(j)  
            ansatz_iris_qmcc += X.on(0, encoder_num - 1)  
            ansatz_iris_qmcc += X.on(1, encoder_num - 1)  
            for k in range(encoder_num - 1):
                if k != encoder_num - 2:
                    ansatz_iris_qmcc += X.on(k + 1, k)
                    ansatz_iris_qmcc += X.on(k + 2, k)

            for j in range(encoder_num):
                ansatz_iris_qmcc += RX(f'gammaRX{j + ansatz_num * i}').on(j)   
            ansatz_iris_qmcc += X.on(0, encoder_num - 1)   
            ansatz_iris_qmcc += X.on(1, encoder_num - 1)  
            for k in range(encoder_num - 1):
                if k != encoder_num - 2:
                    ansatz_iris_qmcc += X.on(k + 1, k)
                    ansatz_iris_qmcc += X.on(k + 2, k)

            for j in range(encoder_num):
                ansatz_iris_qmcc += RZ(f'gammaRZ{j + ansatz_num * i}').on(j)   
            ansatz_iris_qmcc += X.on(0, encoder_num - 1)   
            ansatz_iris_qmcc += X.on(1, encoder_num - 1)   
            for k in range(encoder_num - 1):
                if k != encoder_num - 2:
                    ansatz_iris_qmcc += X.on(k + 1, k)
                    ansatz_iris_qmcc += X.on(k + 2, k)

        circuit_iris_qmcc = encoder_iris_qmcc.as_encoder() + ansatz_iris_qmcc.as_ansatz()
        # print(circuit_iris_qmcc)
        hams_qmcc = [Hamiltonian(QubitOperator(f'Z{i}')) for i in range(ansatz_num)]

        # ansatz_num = encoder_num
        # encoder_iris_qmcc = Circuit()
        # for i in range(encoder_num):
        #     encoder_iris_qmcc += H.on(i)
        # for i in range(encoder_num):
        #     encoder_iris_qmcc += RX(f'alpha{i}').on(i)
        #     encoder_iris_qmcc += RY(f'alpha{i}').on(i)
        #     # encoder_iris_qmcc += RZ(f'alpha{i}').on(i)
        # for i in range(encoder_num):
        #     if i == 0:
        #         encoder_iris_qmcc += X.on(encoder_num-1, 0)
        #         encoder_iris_qmcc += X.on(1, 0)
        #     elif i != encoder_num-1 and i != 0:
        #         encoder_iris_qmcc += X.on(i - 1, i)
        #         encoder_iris_qmcc += X.on(i + 1, i)
        #     elif i == encoder_num-1:
        #         encoder_iris_qmcc += X.on(encoder_num-2, encoder_num-1)
        #         encoder_iris_qmcc += X.on(0, encoder_num-1)
        # encoder_iris_qmcc = encoder_iris_qmcc.no_grad()
        #
        # ansatz_iris_qmcc = Circuit()
        # for i in range(ansatz_num):
        #     for j in range(encoder_num):
        #         ansatz_iris_qmcc += RX(f'betaRX{j + ansatz_num * i}').on(j)
        #         ansatz_iris_qmcc += RY(f'betaRY{j + ansatz_num * i}').on(j)
        #
        #         # ansatz_iris_qmcc += RX(f'gamma{j + ansatz_num * i}').on(j)
        #         # ansatz_iris_qmcc += RY(f'gamma{j + ansatz_num * i}').on(j)
        #
        #     for k in range(encoder_num - 2):
        #         if k == 0:
        #             ansatz_iris_qmcc += X.on(1, 0)   
        #             ansatz_iris_qmcc += X.on(2, 0)   
        #         if k != 0:
        #             ansatz_iris_qmcc += RX(f'betaRX{(k) + ansatz_num * i}').on(k)
        #             ansatz_iris_qmcc += RY(f'betaRY{(k) + ansatz_num * i}').on(k)
        #             ansatz_iris_qmcc += X.on(k+1, k)
        #             ansatz_iris_qmcc += X.on(k+2, k)
        #             # ansatz_iris_qmcc += RY(f'phiRY{(k) + ansatz_num * i}').on(k)
        #             # ansatz_iris_qmcc += X.on(k + 1, k)
        #             # ansatz_iris_qmcc += X.on(k + 2, k)
        #         elif k == encoder_num - 2:
        #             ansatz_iris_qmcc += RX(f'betaRX{(k) + ansatz_num * i}').on(k)
        #             ansatz_iris_qmcc += RY(f'betaRY{(k) + ansatz_num * i}').on(k)
        #             ansatz_iris_qmcc += X.on(k + 1, k)
        #             ansatz_iris_qmcc += X.on(0, k)
        #             # ansatz_iris_qmcc += RY(f'phiRY{(k) + ansatz_num * i}').on(k)
        #             # ansatz_iris_qmcc += X.on(k + 1, k)
        #             # ansatz_iris_qmcc += X.on(0, k)
        #         elif k == encoder_num - 1:
        #             ansatz_iris_qmcc += RX(f'betaRX{(k) + ansatz_num * i}').on(k)
        #             ansatz_iris_qmcc += RY(f'betaRY{(k) + ansatz_num * i}').on(k)
        #             ansatz_iris_qmcc += X.on(0, k)
        #             ansatz_iris_qmcc += X.on(1, k)
        #             # ansatz_iris_qmcc += RY(f'phiRY{(k) + ansatz_num * i}').on(k)
        #             # ansatz_iris_qmcc += X.on(0, k)
        #             # ansatz_iris_qmcc += X.on(1, k)
        #         ansatz_iris_qmcc += RZ(f'betaRZ{(k) + ansatz_num * i}').on(k)
        #         ansatz_iris_qmcc += RY(f'betaRY{(k) + ansatz_num * i}').on(k)
        #         # ansatz_iris_qmcc += RZ(f'phiRZ{(k) + ansatz_num * i}').on(k)
        #         # ansatz_iris_qmcc += RY(f'phiRY{(k) + ansatz_num * i}').on(k)
        #
        #     ansatz_iris_qmcc += RY(f'betaRY{i}').on(k)
        #     ansatz_iris_qmcc += RY(f'thetaRY{i}').on(k)
        # circuit_iris_qmcc = encoder_iris_qmcc.as_encoder() + ansatz_iris_qmcc.as_ansatz()
        # # print(circuit_iris_qmcc)
        # hams_qmcc = [Hamiltonian(QubitOperator(f'Z{i}')) for i in range(encoder_num)]

    grad_ops = Simulator('mqvector', circuit_iris_qmcc.n_qubits).get_expectation_with_grad(hams_qmcc, circuit_iris_qmcc, parallel_worker=n_threads)
    MQ = MQLayer(grad_ops)
    return MQ

class CNNEncoder(nn.Cell):
    def __init__(self, n_input, n_output, kernel_size, stride):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_output, kernel_size=kernel_size, stride=stride, pad_mode='pad')
        self.relu = nn.ReLU()
        self.global_max_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, pad_mode='same')

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.global_max_pool(x)
        return x


class Embedding(tnn.Module):
    def __init__(self, args, Qembed_dim):
        super().__init__()
        self.bert = args.bert_model.to(args.device)
        self.device = args.device
        # self.fc1 = tnn.Linear(768, Qembed_dim).to(args.device)
        self.fc1 = tnn.Linear(768, Qembed_dim).to(args.device)
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, inputs):  # input: [batch_size, seq_len]
        context, mask = inputs[0].to(self.device), inputs[2].to(self.device)
        outputs = self.bert(context, attention_mask=mask)
        hidden_states = outputs.last_hidden_state
        outputs = hidden_states[:, -1, :]
        outputs = ms.Tensor(outputs.detach().cpu().numpy(), dtype=ms.float32)
        # outputs = self.fc1(outputs)
        return outputs


class RNN(nn.Cell):
    def __init__(self, hidden_dim, output_dim, n_layers, args):
        super().__init__()
        self.bert = args.bert_model.to(args.device)
        self.Qembed_num = 11
        self.Qbits = args.qbits
        self.embedding = Embedding(args, self.Qembed_num)
        self.qembedding1 = QEmbedding1(self.Qbits, args.n_class, args.withlabel)
        # self.qembedding2 = QEmbedding2(self.Qbits, args.n_class, args.withlabel)
        self.withlabel = args.withlabel

        self.device = args.device
        self.fc = nn.Dense(768, self.Qbits)
        # self.fc = nn.Dense(768, self.Qbits, weight_init=init.Normal(sigma=0.01), bias_init=init.Zero())
        # self.fc1 = nn.Dense(768, self.Qbits, weight_init=init.XavierUniform(), bias_init=init.Constant(value=0.1))
        # self.cast = ops.Cast()
        self.dense1 = nn.Dense(11, output_dim)
        # self.dense2 = nn.Dense(768, args.n_class)
        # self.relu = ops.ReLU()
        self.softmax = nn.Softmax(axis=-1)
        # self.cnn_encoder1 = CNNEncoder(1, 2, kernel_size=5, stride=1)
        # self.cnn_encoder2 = CNNEncoder(1, 2, kernel_size=4, stride=1)
        # self.cnn_encoder3 = CNNEncoder(1, 2, kernel_size=3, stride=1)
        for param in self.bert.parameters():
            param.requires_grad = False

    def construct(self, inputs):  # input: [batch_size, seq_len]
        outputs = self.embedding(inputs)
        outputs0 = self.fc(outputs)
        # outputs1 = self.fc1(outputs)

        reduce_min = ops.ReduceMin(keep_dims=True)
        reduce_max = ops.ReduceMax(keep_dims=True)
        min_vals0 = reduce_min(outputs0, 1)
        max_vals0 = reduce_max(outputs0, 1)
        normalized_tensor0 = (outputs0 - min_vals0) * 10 / (max_vals0 - min_vals0 + 1e-9)
        # normalized_tensor0 = normalized_tensor0 * 2 - 10
        # min_vals1 = reduce_min(outputs1, 1)
        # max_vals1 = reduce_max(outputs1, 1)
        # normalized_tensor1 = (outputs1 - min_vals1) / (max_vals1 - min_vals1 + 1e-9)
        embedded0 = self.qembedding1(normalized_tensor0)
        # embedded1 = self.qembedding2(normalized_tensor1)
        # embedded = embedded.expand_dims(1)

        # concat_op = P.Concat(axis=-1)
        # cnn_output = concat_op((embedded0, embedded1))
        # embedded = embedded0.expand_dims(1)  # [batch_size, 1, n]
        # cnn_output = self.cnn_encoder2(embedded)
        # global_max_pool_output = cnn_output.max(axis=2)
        # global_max_pool_output = self.softmax(global_max_pool_output)
        # global_max_pool_output = self.dense1(global_max_pool_output)
        # result_tensor = self.softmax(global_max_pool_output)

        if self.withlabel == 'no':
            result_tensor = self.dense1(embedded0)
        #     result_tensor = self.dense2(result_tensor)
        # result_tensor = self.softmax(result_tensor)
        # return global_max_pool_output
        return result_tensor


# class Bert(tnn.Module):
#     def __init__(self, args):
#         super(Bert, self).__init__()
#         self.bert = args.teacher_model.to(args.device)
#         # self.fc1 = tnn.Linear(1024, args.n_class)
#         self.fc1 = tnn.Linear(1024, 2)
#         # self.fc1 = tnn.Linear(768, args.n_class)
#         self.dropout = tnn.Dropout(0.5)
#         self.device = args.device
#         for param in self.bert.parameters():
#             param.requires_grad = False
#
#     def forward(self, x):
#         context, mask = x[0].to(self.device), x[2].to(self.device)
#         # -------------- bert --------------
#         # _, outputs = self.bert(context, attention_mask=mask)
#         # -------------- gpt --------------
#         outputs = self.bert(context, attention_mask=mask)
#         outputs = outputs[0]
#         outputs = outputs[:, -1, :]
#         outputs = self.fc1(outputs)
#         return outputs


# class LLM_teacher(tnn.Module):
#     def __init__(self, base_model_path, tokenizer_path, lora_weights, template_path):
#         super(LLM_teacher, self).__init__()
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#         
#         self.model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map='auto', torch_dtype=torch.float16, load_in_8bit=True)
#         self.model = PeftModel.from_pretrained(self.model, lora_weights, device_map=self.device)
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
#         self.tokenizer.pad_token_id = self.tokenizer.unk_token_id = 0
#         self.tokenizer.padding_side = "left"
#
#         
#         with open(template_path) as fp:
#             self.template = json.load(fp)
#
#     def forward(self, texts):
#         data = [{"instruction": text, "input": "", "output": ""} for text in texts]
#         prompt = []
#         for i in data:
#             if i['input']:
#                 prompt.append(self.template['prompt_input'].format(instruction=i['instruction'], input=i['input']))
#             else:
#                 prompt.append(self.template['prompt_no_input'].format(instruction=i['instruction']))
#
#         
#         batch_tokenizer = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
#         input_ids = batch_tokenizer["input_ids"]
#
#         with torch.no_grad():
#             generation_output = self.model(input_ids)
#             log_prob = generation_output.logits
#             prob = torch.softmax(log_prob, dim=-1)
#             log_prob = log_prob.view(prob.size(0), -1, prob.size(-1))[:, -1, :]
#             # prob = prob / prob.sum()
#             indices_to_select = [8178, 6374]
#             device = prob.device
#             indices_tensor = torch.tensor(indices_to_select, device=device)
#             selected_prob = torch.index_select(log_prob, dim=1, index=indices_tensor)
#             # selected_prob = torch.softmax(selected_prob, dim=-1)
#
#         return selected_prob

class suppress_stdout_stderr(object):
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
        os.close(self.save_fds[0])
        os.close(self.save_fds[1])


def normalize_to_neg1_pos1(arr):
    min_val = torch.min(arr)
    max_val = torch.max(arr)
    normalized_data = 2 * ((arr - min_val) / (max_val - min_val)) - 1
    return normalized_data


class LLM_teacher_gen(tnn.Module):
    def __init__(self, base_model_path, tokenizer_path, lora_weights, template_path, args):
        super(LLM_teacher_gen, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.n_class = args.n_class
        # Load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map='auto', torch_dtype=torch.float16, load_in_8bit=True)
        self.model = PeftModel.from_pretrained(self.model, lora_weights, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        self.tokenizer.pad_token_id = self.tokenizer.unk_token_id = 0
        self.LLM = args.LLM
        if self.LLM == 'Baichuan2-7B':
            self.tokenizer.padding_side = "right"
        else:
            self.tokenizer.padding_side = "left"

        # Load the template
        with open(template_path) as fp:
            self.template = json.load(fp)

    def save_and_load_data(self, texts):
        # Prepare data
        data = [{"instruction": text, "input": "", "output": ""} for text in texts]
        json_path = './instructions.json'

        # Save data to JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        with suppress_stdout_stderr():
            dataset = load_dataset('json', data_files=json_path, split='train')

        # Optionally remove the file after loading
        if os.path.exists(json_path):
            os.remove(json_path)

        return dataset

    def forward(self, texts):
        dataset = self.save_and_load_data(texts)
        prompts = []
        for item in dataset:
            if item['input']:
                prompts.append(self.template['prompt_input'].format(instruction=item['instruction'], input=item['input']))
            else:
                prompts.append(self.template['prompt_no_input'].format(instruction=item['instruction']))

        # Process input data
        batch_tokenizer = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        input_ids = batch_tokenizer["input_ids"]

        with torch.no_grad():
            generation_output = self.model(input_ids)
            log_prob = generation_output.logits
            prob = torch.softmax(log_prob, dim=-1)
            log_prob = log_prob.view(prob.size(0), -1, prob.size(-1))[:, -1, :]
            # log_prob = log_prob / log_prob.sum()
            if self.LLM in ('LLaMA2-7B', 'LLaMA2-13B'):
                if self.n_class == 2:
                    indices_to_select = [8178, 6374]
                elif self.n_class == 3:
                    indices_to_select = [26277, 28503, 8178]
                elif self.n_class == 4:
                    indices_to_select = [5381, 9327, 14717, 3186]
            elif self.LLM in ('BLOOMZ-1.1B', 'BLOOMZ-3B'):
                if self.n_class == 2:
                    indices_to_select = [111017, 96675]
                elif self.n_class == 3:
                    indices_to_select = [0, 0, 111017]
                elif self.n_class == 4:
                    indices_to_select = [113782, 199808, 0, 42199]
            elif self.LLM == 'LLaMA3-8B':
                if self.n_class == 2:
                    indices_to_select = [43324, 31587]
                elif self.n_class == 3:
                    indices_to_select = [0, 21648, 111017]
                elif self.n_class == 4:
                    indices_to_select = [113782, 199808, 0, 42199]

            # log_prob = normalize_to_neg1_pos1(log_prob)
            indices_tensor = torch.tensor(indices_to_select, device=self.device)
            selected_prob = torch.index_select(log_prob, dim=1, index=indices_tensor)
            # selected_prob = normalize_to_neg1_pos1(selected_prob)
            # selected_prob = torch.softmax(selected_prob, dim=-1)

        return selected_prob

class LLM_teacher(tnn.Module):
    def __init__(self, base_model_path, tokenizer_path, lora_weights, template_path, args):
        super(LLM_teacher, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.n_class = args.n_class
        # self.model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map='auto',
        #                                                   torch_dtype=torch.float16, load_in_8bit=True)
        # self.model = PeftModel.from_pretrained(self.model, lora_weights, device_map=self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        self.tokenizer.pad_token_id = self.tokenizer.unk_token_id = 0
        self.LLM = args.LLM
        if self.LLM == 'Baichuan2-7B':
            # self.tokenizer.padding_side = "right"
            self.model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=base_model_path, trust_remote_code=True,
                device_map='auto', num_labels=2, torch_dtype=torch.float16)
        else:
            # self.tokenizer.padding_side = "left"
            self.model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=base_model_path,
                load_in_8bit=True, device_map='auto', trust_remote_code=True,
                num_labels=4, torch_dtype=torch.float16)
        self.tokenizer.padding_side = "right"

        self.model = PeftModel.from_pretrained(self.model, lora_weights,
                                               device_map=self.device, torch_dtype=torch.float16)
        self.model.config.pad_token_id = self.model.config.unk_token_id = 0
        # Load the template
        with open(template_path) as fp:
            self.template = json.load(fp)

    def save_and_load_data(self, texts):
        data = [{"instruction": text, "input": "", "output": ""} for text in texts]
        json_path = './instructions.json'

        # Save data to JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        with suppress_stdout_stderr():
            dataset = load_dataset('json', data_files=json_path, split='train')

        # Optionally remove the file after loading
        if os.path.exists(json_path):
            os.remove(json_path)

        return dataset

    def forward(self, texts):
        dataset = self.save_and_load_data(texts)
        prompts = []
        label = []
        for i in dataset:
            if i['input']:
                prompts.append(self.template['prompt_input'].format(instruction=i['instruction'], input=i['input']))
            else:
                prompts.append(self.template['prompt_no_input'].format(instruction=i['instruction']))
            label.append(i['output'])

        batch_texts = prompts
        batch_tokenizer = self.tokenizer(batch_texts, return_tensors="pt", padding=True).to(self.device)
        output = self.model(**batch_tokenizer).logits
        return output
