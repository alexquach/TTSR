from torch.utils.data import DataLoader
from importlib import import_module


image_dir = "./Data_CDVL_LR_MC_uf_frames4_ps_72_fn_6_tpn_1000.h5"
ref_dir = "./Data_CDVL_HR_uf_frames4_ps_72_fn_6_tpn_1000.h5"


def get_dataloader(args):
    ### import module
    m = import_module('dataset.' + args.dataset.lower())

    if (args.dataset == 'HMDB_FRAMES'):
        # (self, image_dataset_dir, ref_dataset_dir, upscale_factor, input_transform=None, ref_transform=None)
        data_train = getattr(m, 'TrainSet')(args)
        print(data_train)
        dataloader_train = DataLoader(
            data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloader_test = {}
        for i in range(5):
            data_test = getattr(m, 'TestSet')(args=args)
            dataloader_test[str(i+1)] = DataLoader(data_test, batch_size=1,
                                                   shuffle=False, num_workers=args.num_workers)
        dataloader = {'train': dataloader_train, 'test': dataloader_test}

    elif (args.dataset == 'HMDB_FLOWNET'):
        # (self, image_dataset_dir, ref_dataset_dir, upscale_factor, input_transform=None, ref_transform=None)
        data_train, data_test = m.get_train_test_sets(args, train_test_split=0.8)
        
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloader_test = {}
        dataloader_test['1'] = DataLoader(data_test, batch_size=1,
                                                shuffle=False, num_workers=args.num_workers)
        dataloader = {'train': dataloader_train, 'test': dataloader_test}

    else:
        raise SystemExit('Error: no such type of dataset!')

    return dataloader
