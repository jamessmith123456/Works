#encoding=utf-8
import os
import numpy as np
import pickle
import xlrd

def read_fasta(raw_fasta_path, rawpath):
    path = os.path.join(raw_fasta_path, rawpath)
    #path = 'F:/DrugScreening2/mydata/targets_drug/fasta/O14925.fasta'
    with open(path,'r') as f:
        content = f.readlines()
    drop_head_content = content[1:]
    result = ''
    for i in range(len(drop_head_content)):
        result += drop_head_content[i].strip('\n')
    result = result.replace('\n','')
    return result

def rewrite_fasta(root_path, file_name, save_root_path):
    #path = 'F:/DrugScreening2/mydata/targets_drug/fasta/O14925.fasta'
    with open(os.path.join(root_path, file_name),'r') as f:
        content = f.readlines()
    head = '>'+content[0].split('|')[1]
    drop_head_content = content[1:]
    result = ''
    for i in range(len(drop_head_content)):
        result += drop_head_content[i].strip('\n')
    result = result.replace('\n','')
    with open(os.path.join(save_root_path, file_name),'w') as f:
        f.writelines(head)
        f.writelines('\n')
        f.writelines(result)
    f.close()
    return result

def read_pssm(pssm_root_path, file):
    path = os.path.join(pssm_root_path, file+'/logpssm/'+file+'.pssm')
    #file = 'F:/DrugScreening2/mydata2/targets_drug/pssm/O00400/logpssm/O00400.pssm'
    with open(path,'r') as f:
        content = f.readlines()
    pssm = []
    for i in range(len(content)):
        temp_pssm = []
        temp_pssm_str = content[i].split('    ')
        for j in range(1, len(temp_pssm_str)-1):
            temp_pssm.append(float(temp_pssm_str[j]))
        pssm.append(temp_pssm)

    # pse_pssm = []
    # for i in range(len(pssm)):
    #     pse_pssm.append([np.mean(pssm[i]), np.std(pssm[i])])
    # print("file:{}".format(len(pssm)))
    # print("file:{}".format(len(pse_pssm)))
    return pssm

def read_pss(pss_root_path, file):
    path = os.path.join(pss_root_path, file + '/ss/' + file + '.ss')
    with open(path,'r') as f:
        content = f.readlines()
    pss = []
    for i in range(len(content)):
        temp_pss = []
        temp_pss_str = content[i].split('    ')
        for j in range(1, len(temp_pss_str)):
            temp_pss.append(float(temp_pss_str[j].strip('\n')))
        pss.append(temp_pss)
    print("file:{}".format(len(pss)))
    return pss

def get_drug_target(root_path, name):
    path = os.path.join(root_path,name+'.xlsx')
    s = xlrd.open_workbook(path)
    sheet = s.sheet_by_index(0)
    #print('nrows:',sheet.nrows)
    #print('ncols:',sheet.ncols)
    five_target_gene = []
    five_target_gene_effect_size = []
    five_target_gene_p_value = []
    for col_index in range(1,6):
        five_target_gene.append(sheet.cell(0, col_index).value)
        five_target_gene_effect_size.append(sheet.cell(1, col_index).value)
        five_target_gene_p_value.append(sheet.cell(2, col_index).value)
    return five_target_gene, five_target_gene_effect_size, five_target_gene_p_value

def get_gene_fasta(root_path, file):
    # root_path = 'F:/DrugScreening2/mydata2/targets_drug'
    # file = 'gene_fasta.xlsx'
    path = os.path.join(root_path, file)
    s = xlrd.open_workbook(path)
    sheet = s.sheet_by_index(0)
    gene_fasta = {}
    for i in range(sheet.nrows):
        if sheet.cell_value(i,0)!='' and sheet.cell_value(i,1)!='':
            gene_fasta[sheet.cell_value(i,0)] = sheet.cell_value(i,1).split('.')[0]
    return gene_fasta

if __name__ == '__main__':
    # raw_fasta_path = 'F:/DrugScreening2/mydata2/targets_drug/fasta'
    # all_fasta_file = os.listdir(raw_fasta_path)
    # save_fasta_path = 'F:/DrugScreening2/mydata2/targets_drug/fasta2'
    # for item in all_fasta_file:
    #     rewrite_fasta(raw_fasta_path, item, save_fasta_path)
    #
    # all_fasta_seq = {}
    # for item in all_fasta_file:
    #     all_fasta_seq[item] = read_fasta(raw_fasta_path, item)
    # all_length = []
    # for item in all_fasta_file:
    #     if '-' not in item:
    #         all_length.append(len(all_fasta_seq[item]))
    # #现在F:/DrugScreening2/mydata2/targets_drug/fasta2路径下全是序列的fasta文件 然后使用教研室服务器上的
    # pssm_root_path = 'F:/DrugScreening2/mydata2/targets_drug/pssm'
    # all_fasta_pssm_file = os.listdir(pssm_root_path)
    # all_fasta_pssm_file_sub = []
    # for item in all_fasta_pssm_file:
    #     if '-' not in item:
    #         all_fasta_pssm_file_sub.append(item) #72个
    # all_fasta_pssm = {}
    # for item in all_fasta_pssm_file_sub:
    #     all_fasta_pssm[item] = read_pssm(pssm_root_path, item)
    #
    # pss_root_path = 'F:/DrugScreening2/mydata2/targets_drug/pss'
    # all_fasta_pss_file = os.listdir(pss_root_path)
    # all_fasta_pss_file_sub = []
    # for item in all_fasta_pss_file:
    #     if '-' not in item:
    #         all_fasta_pss_file_sub.append(item) #72个
    # all_fasta_pss = {}
    # for item in all_fasta_pss_file_sub:
    #     all_fasta_pss[item] = read_pss(pss_root_path, item)
    #
    # if all_fasta_pssm_file_sub==all_fasta_pss_file_sub:
    #     print("pssm num MATCH pss mun!")
    # else:
    #     print("pssm num NOT MATCH pss mun!")
    #
    # all_fasta = all_fasta_pssm_file_sub
    # all_fasta_pssm_pss_feature = {}
    # for item in all_fasta:
    #     temp_pssm = all_fasta_pssm[item]
    #     temp_pss = all_fasta_pss[item]
    #     for j in range(len(temp_pssm)):
    #         temp_pssm[j].extend(temp_pss[j])
    #     print("len(temp_pssm):",len(temp_pssm))
    #     all_fasta_pssm_pss_feature[item] = temp_pssm
    #
    # max_length = 0
    # for item in all_fasta_pssm_pss_feature.keys():
    #     all_fasta_pssm_pss_feature[item] = np.array(all_fasta_pssm_pss_feature[item])
    #     if all_fasta_pssm_pss_feature[item].shape[0]>max_length:
    #         max_length = all_fasta_pssm_pss_feature[item].shape[0]
    #
    # for item in all_fasta_pssm_pss_feature.keys():
    #     new_array = np.zeros((max_length, 23))
    #     before_array = all_fasta_pssm_pss_feature[item]
    #     for i in range(before_array.shape[0]):
    #         new_array[i,:] = before_array[i,:]
    #     all_fasta_pssm_pss_feature[item] = new_array.transpose(1,0)
    # save_file = open("F:/DrugScreening2/mydata2/targets_drug/feature_pkl/gene_psepssm_pss_feature.pkl",'wb')
    # pickle.dump(all_fasta, save_file)
    # pickle.dump(all_fasta_pssm_pss_feature, save_file)
    # save_file.close()
    #
    # # load_file = open("F:/DrugScreening2/mydata2/targets_drug/feature_pkl/gene_pssm_pss_feature.pkl",'rb')
    # # all_fasta = pickle.load(load_file) #一共72个gene名字
    # # all_fasta_pssm_pss_feature = pickle.load(load_file) #每个gene所对应的23*1279的特征  如果是psepssm 可能会更好？
    # # load_file.close()
    #
    # root_path = 'F:/DrugScreening2/mydata2/targets_drug/cell_line'
    # all_cellline_xlsx = os.listdir(root_path)
    # all_cellline = [item.split('.')[0] for item in all_cellline_xlsx]
    #
    # all_cellline_5_gene_info = {}
    # for item in all_cellline:
    #     all_cellline_5_gene_info[item] = get_drug_target(root_path, item)
    #
    # root_path = 'F:/DrugScreening2/mydata2/targets_drug'
    # file = 'gene_fasta.xlsx'
    # gene_fasta = get_gene_fasta(root_path, file)
    #
    # all_cellline_feature = {}
    # for item in all_cellline:
    #     pssm_pss = np.zeros((5*5, max_length))
    #     for j in range(len(all_cellline_5_gene_info[item][0])):
    #         pssm_pss[j*5:(j+1)*5,:] = all_fasta_pssm_pss_feature[gene_fasta[all_cellline_5_gene_info[item][0][j]]]
    #     all_cellline_feature[item] = pssm_pss
    # save_file = open("F:/DrugScreening2/mydata2/targets_drug/feature_pkl/cellline_psepssm_pss_feature.pkl",'wb')
    # pickle.dump(all_cellline, save_file)
    # pickle.dump(all_cellline_feature, save_file)
    # save_file.close()
    #
    # # load_file = open("F:/DrugScreening2/mydata2/targets_drug/feature_pkl/cellline_pssm_pss_feature.pkl",'rb')
    # # all_cellline = pickle.load(load_file) #21种癌细胞系
    # # all_cellline_feature = pickle.load(load_file) #每种癌细胞系为115*1279的特征
    # # load_file.close()
    #
    #
    # #0719再次使用
    # # raw_fasta_path = 'C:/Users/Administrator/Desktop/database/mydatabase/all_fasta'
    # # all_fasta_file = os.listdir(raw_fasta_path)
    # # save_fasta_path = 'C:/Users/Administrator/Desktop/database/mydatabase/all_fasta2'
    # # for item in all_fasta_file:
    # #     rewrite_fasta(raw_fasta_path, item, save_fasta_path)
    # # 现在F:/DrugScreening2/mydata2/targets_drug/fasta2路径下全是序列的fasta文件 然后使用教研室服务器上的
    #pssm_root_path = 'C:/Users/Administrator/Desktop/database/mydatabase/pssm'
    pssm_root_path = './pssm'
    all_fasta_pssm_file = os.listdir(pssm_root_path)
    all_fasta_pssm_file_sub = []
    for item in all_fasta_pssm_file:
        if '-' not in item:
            all_fasta_pssm_file_sub.append(item)  # 72个
    all_fasta_pssm = {}
    for item in all_fasta_pssm_file_sub:
        all_fasta_pssm[item] = read_pssm(pssm_root_path, item) #233个

    #pss_root_path = 'C:/Users/Administrator/Desktop/database/mydatabase/pss'
    pss_root_path = './pss'
    all_fasta_pss = {}
    for item in all_fasta_pssm_file_sub:
        all_fasta_pss[item] = read_pss(pss_root_path, item)

    #all_fasta_pss 和 all_fasta_pssm都是包含N个元素的list 每个list的长度为20
    all_fasta = all_fasta_pssm_file_sub
    all_fasta_pssm_pss_feature = {}
    for item in all_fasta:
        temp_pssm = all_fasta_pssm[item]
        temp_pss = all_fasta_pss[item]
        for j in range(len(temp_pssm)):
            temp_pssm[j].extend(temp_pss[j])
        print("len(temp_pssm):",np.array(temp_pssm).shape)
        all_fasta_pssm_pss_feature[item] = temp_pssm #all_fasta_pssm_pss_feature是个字典 每个value是一个list 长度为N 每个元素是23维

    max_length = 1279

    for item in all_fasta_pssm_pss_feature.keys():
        new_array = np.zeros((max_length, 23))
        before_array = all_fasta_pssm_pss_feature[item]
        before_array = np.array(before_array)
        for i in range(min(before_array.shape[0],max_length)):
            new_array[i,:] = before_array[i,:] #(1279,23)
        all_fasta_pssm_pss_feature[item] = new_array.transpose(1,0) #(23,1279)
    save_file = open("./gene_psepssm_pss_feature.pkl",'wb')
    pickle.dump(all_fasta, save_file)
    pickle.dump(all_fasta_pssm_pss_feature, save_file)
    save_file.close()

    import pickle
    #load_file = open("C:/Users/Administrator/Desktop/database/mydatabase/Establish_database.pkl","rb")
    load_file = open("./Establish_database.pkl", "rb")
    all_cell_line_genes = pickle.load(load_file) #1965
    all_gene_name = pickle.load(load_file) #1605
    all_uniprot_id = pickle.load(load_file) #1605
    all_href = pickle.load(load_file) #一共235个链接
    load_file.close()

    all_cellline_feature = {}
    for key,value in all_cell_line_genes.items():
        if value!=[]:
            pssm_pss = np.zeros((5*23, max_length))
            for j in range(5):
                temp_gene = value[j]
                print("cell_line:{}/gene{}:{}".format(key, j, temp_gene))
                pssm_pss[j*23:(j+1)*23,:] = all_fasta_pssm_pss_feature[all_uniprot_id[all_gene_name.index(temp_gene)]]
            pssm_pss = pssm_pss[np.newaxis, :,:]
            all_cellline_feature[key] = pssm_pss

    save_file = open('cell_line_genes_GEBF.pkl','wb')
    pickle.dump(all_cellline_feature, save_file)
    save_file.close()
    print("all_cellline_feature.keys()",len(all_cellline_feature.keys()))
    for item in all_cellline_feature.keys():
        print(all_cellline_feature[item].shape)




