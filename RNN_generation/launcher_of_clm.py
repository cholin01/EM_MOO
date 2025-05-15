import torch
import torch.optim as optim

import configs as cfg
from utils import builder, smi_tools, tools
from models import clm_rnn
from rdkit import Chem

def train_clm(data_path: str, smi_idx: int, model_name='clm', epochs=100, fq_saving=5,
              model_path=None, pre_data_path=None, pre_smi_idx=None, fine_tuning=False):

    # processing smiles
    data, _ = tools.load_data_from_csv(data_path, with_head=True)
    smiles = [cfg.BOS + x[smi_idx] + cfg.EOS for x in data]
    tokens = smi_tools.gather_tokens(smiles, single_split=cfg.SINGLE_TOKENIZE)
    print(f'Tokens: {tokens}')

    #
    if pre_data_path:
        data, _ = tools.load_data_from_csv(pre_data_path, with_head=True)
        pre_smiles = [cfg.BOS + x[pre_smi_idx] + cfg.EOS for x in data]
        tokens = smi_tools.gather_tokens(pre_smiles, single_split=cfg.SINGLE_TOKENIZE)
        print(len(data))
        print(f'Tokens of pre-trained data: {tokens}')
        print(f'There are {len(smiles)} SMILES strings in data.')
        smiles = [smi for smi in smiles if smi_tools.if_oov_exclude(smi, tokens, single_split=cfg.SINGLE_TOKENIZE)]
        print(f'There are {len(smiles)} SMILES strings after checking tokens.')

    loader = builder.clm_packer(smiles, tokens)

    # initialize clm
    print('len of tokens:{}'.format(len(tokens)))
    m = builder.build_clm(len(tokens), model_path)
    # initial optimizer
    if fine_tuning:
        opt = optim.Adam(m.parameters_to_train(), lr=cfg.CLM_LR_RATE)
    else:
        opt = optim.Adam(m.parameters(), lr=cfg.CLM_LR_RATE)

    # training
    records = clm_rnn.train(model=m, optimizer=opt, data_loader=loader,
                            epochs=epochs, fq_of_save=fq_saving, name=model_name)

    return 0

def generate(n: int, idx: int, data_path: str, origin_path, model_path: str, saving_path: str) -> list:
    # processing smiles
    ori_data, ori_head = tools.load_data_from_csv(origin_path, with_head=True)
    data, head = tools.load_data_from_csv(data_path, with_head=True)
    smiles = [x[idx] for x in ori_data]
    raw_smiles = [x[idx] for x in data]
    # raw_smiles = [smi_tools.to_canonical_smi(smi) for smi in smiles]
    raw_smiles = [smi for smi in raw_smiles if smi is not None]

    # initialize clm
    tokens = smi_tools.gather_tokens([cfg.BOS + smi + cfg.EOS for smi in smiles], single_split=cfg.SINGLE_TOKENIZE)
    #print(len(tokens))
    m = builder.build_clm(len(tokens), model_path)

    # sampling
    print('Sampling ...')
    novel_smiles, record = builder.generate(n, m, raw_smiles, tokens)
    print('Sampling Finished !')
    print(f'Sample:{n}, Valid:{record[1]}, Unique:{record[2]}, Novel:{record[3]}')
    tools.save_data_to_csv(saving_path, [[smi] for smi in novel_smiles], ['smiles'])

    return novel_smiles


if __name__ == '__main__':
   generate(50, 0, './data/emsmiles.csv' , './data/guacamol_train.csv', './record/gutl-0050-12.2942.pth', 'generate_mol.csv')