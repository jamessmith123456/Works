import sys
import argparse
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import DataStructs
import pickle
import numpy as np
import os
from shutil import copyfile
import zipfile
import smtplib
import mimetypes,os
from email.mime.base import MIMEBase
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib,time,os,zipfile,mimetypes
from email.header import Header
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Get SMILE-mol from SMILE-file
def get_SMILE(drug_file):
    with open(drug_file,'r') as f:
        smile = f.readlines()
    return smile[0]

#get MACCSKeys
def get_MACCSKeys(smile):
    mol = Chem.MolFromSmiles(smile)
    fp1 = MACCSkeys.GenMACCSKeys(mol)
    fp1 = fp1.ToBitString()
    result = []
    for i in range(len(fp1)):
        result.append(float(fp1[i]))
    return np.array(result)[np.newaxis,:]

#get MorganFingerprints
def get_MORGANKeys(smile):
    mol = Chem.MolFromSmiles(smile)
    fp1_morgan = AllChem.GetMorganFingerprint(mol, 2)
    fp1_morgan_hashed = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp1_morgan_hashed = fp1_morgan_hashed.ToBitString()  # 一个长为1024由0-1组成的字符串
    result = []
    for i in range(len(fp1_morgan_hashed)):
        result.append(float(fp1_morgan_hashed[i]))
    return np.array(result)[np.newaxis,:]

#get CellLine-GEBF
def get_CellLine(cell_line):
    #load_file = open('./myweb/cell_line_genes_GEBF.pkl','rb')
    load_file = open('./cell_line_genes_GEBF.pkl', 'rb')
    all_cellline_pssm_pss = pickle.load(load_file)
    load_file.close()
    return all_cellline_pssm_pss[cell_line]

#get five genes-related cell-line
def get_five_genes_fasta(cell_line):
    #load_file = open('./myweb/cell_line_genes.pkl','rb') #一共321种癌细胞系
    load_file = open('./cell_line_genes.pkl', 'rb')  # 一共321种癌细胞系
    cell_line_genes = pickle.load(load_file)
    load_file.close()
    return cell_line_genes[cell_line]

#zip
def make_zip(source_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)   #相对路径
            zipf.write(pathfile, arcname)
    zipf.close()

#Send email
class SendEmail():
    """发送邮件"""
    def __init__(self, host, port, username, password, receivername, zipfile):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.receivername = receivername
        self.zipfile = zipfile

    def annex(self):
        """将zip添加附件"""
        try:
            data = open(self.zipfile, 'rb')
            ctype, encoding = mimetypes.guess_type(self.zipfile)
            if ctype is None or encoding is not None:
                ctype = 'application/x-zip-compressed'
            maintype, subtype = ctype.split('/', 1)
            file_msg = MIMEBase(maintype, subtype)
            file_msg.set_payload(data.read())
            data.close()
            encoders.encode_base64(file_msg)
            basename = os.path.basename(self.zipfile)
            file_msg.add_header('Content-Disposition', 'attachment', filename=basename)
            return file_msg
        except:
            print('文件添加失败')
            raise

    def data(self):
        """邮件内容"""
        try:
            message_annex = MIMEMultipart()
            annex = self.annex()
            # """构建根容器"""
            test = MIMEText('This email comes from the GEDCNN, which is an online service for predicting drug-drug-cancer cell line synergies; no reply required.', 'plain', 'utf-8')
            message_annex['From'] = Header("GEDCNN", 'utf-8')  # 发送者
            message_annex['To'] = Header(self.receivername, 'utf-8')  # 接收者
            subject = 'GEDCNN Web Server Process Result.'
            message_annex['Subject'] = Header(subject, 'utf-8')
            message_annex.attach(annex)
            message_annex.attach(test)
            #fullTest = message_annex.as_string()
            fullTest = message_annex.as_string()
            return fullTest
        except:
            raise

    def send_eamil(self):
        message = self.data()
        smtpObj = smtplib.SMTP_SSL(self.host, self.port)
        smtpObj.login(self.username, self.password)
        smtpObj.sendmail(self.username, self.receivername, message)
        smtpObj.quit()

class Server():
    def __init__(self, SMILE_A, SMILE_B, cell_line):
        self.SMILE_A = SMILE_A
        self.SMILE_B = SMILE_B
        self.cell_line = cell_line
        self.GEBF = get_CellLine(self.cell_line)
        self.MACCKEYS_A = get_MACCSKeys(self.SMILE_A)
        self.MACCKEYS_B = get_MACCSKeys(self.SMILE_B)
        print("input_x1:", self.MACCKEYS_A.shape)
        print("input_x1:", self.MACCKEYS_B.shape)
        print("input_x1:", self.GEBF.shape)

    def predict(self):
        input1 = self.MACCKEYS_A
        input2 = self.MACCKEYS_B
        input3 = self.GEBF
        sess = tf.Session()
        with gfile.FastGFile('./modelpb/model.pb', 'rb') as f:
            with tf.device('/cpu:0'):
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='') # 导入计算图
            sess.run(tf.global_variables_initializer())
            input_x1 = sess.graph.get_tensor_by_name('input_x1:0')
            input_x2 = sess.graph.get_tensor_by_name('input_x2:0')
            input_x3 = sess.graph.get_tensor_by_name('input_x3:0')
            op = sess.graph.get_tensor_by_name('logits_to_store:0')
            ret = sess.run(op, feed_dict={input_x1: input1, input_x2: input2, input_x3: input3})
            if ret[0]<ret[1]:
                self.result = 0
                #print('0')
            else:
                self.result = 1
                print('1')
            print(1/(1+math.exp(max(ret))))

    def store_zip(self):
        self.predict()

        root_path = './GEDCNN_Email_Result'
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        with open(os.path.join(root_path,'drugA.smiles'), 'w') as f:
            f.writelines(self.SMILE_A)
        with open(os.path.join(root_path,'drugB.smiles'), 'w') as f:
            f.writelines(self.SMILE_B)
        save_file = open(os.path.join(root_path,'drug_cancer_line.pkl'),'wb')
        pickle.dump(self.MACCKEYS_A, save_file)
        pickle.dump(self.MACCKEYS_B, save_file)
        pickle.dump(self.GEBF, save_file)
        pickle.dump(self.result, save_file)
        save_file.close()

        self.five_genes = get_five_genes_fasta(self.cell_line)
        for item in self.five_genes:
            copyfile(os.path.join('./database/cell_lines_genes/', item+'.fasta'), os.path.join(root_path, item+'.fasta'))
        make_zip(root_path, './GEDCNN_Email_Result.zip')

        if os.path.exists(root_path):
            os.rmdir(root_path)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Reading parameters from Java")
    #
    # parser.add_argument('--drugA', type=str, default='FC1=CNC(=O)NC1=O', metavar='N', help='drugA_smile_expression')
    # parser.add_argument('--drugB', type=str, default='FC1=CNC(=O)NC1=O', metavar='N', help='drugB_smile_expression')
    # parser.add_argument('--CancerCellLineName', type=str, default='A2058', metavar='N', help='Cancer_CellLine_Name')
    # parser.add_argument('--ReceiverEmail', type=str, default='1321985517@qq.com', metavar='N', help='Receiver_email_address')
    #
    # args = parser.parse_args()
    # drugA_smile = args.drugA
    # drugB_smile = args.drugB
    # cancer_line = args.CancerCellLineName
    # ReceiverEmail = args.ReceiverEmail

    # drugA_smile = sys.argv[1]
    # drugB_smile = sys.argv[2]
    # cancer_line = sys.argv[3]
    # ReceiverEmail = sys.argv[4]

    drugA_smile = 'FC1=CNC(=O)NC1=O'
    drugB_smile = 'FC1=CNC(=O)NC1=O'
    cancer_line = 'SIDM00797' #'A2058'
    ReceiverEmail = '1321985517@qq.com'
    #print("arg1:{},arg2:{},arg3:{},arg4:{}".format(drugA_smile, drugB_smile, cancer_line, ReceiverEmail))
    server = Server(drugA_smile, drugB_smile, cancer_line)
    server.store_zip()

    #F:/DrugScreening2/GEDCNN_Email_Result.zip
    email = SendEmail('smtp.qq.com', 465, '3216370643@qq.com', 'nfsuacwieuocdcdf', ReceiverEmail,'GEDCNN_Email_Result.zip')
    email.send_eamil()

    #python -u webserver.py FC1=CNC(=O)NC1=O FC1=CNC(=O)NC1=O A2058 1321985517@qq.com

    #python -u webserver.py --drugA 'FC1=CNC(=O)NC1=O' --drugB 'FC1=CNC(=O)NC1=O' --CancerCellLineName 'A2058' --ReveiverEmail '1321985517@qq.com'

    #
    # all_cell_line_five_genes = {}
    # for key,value in all_cell_line_genes.items():
    #     if value!=[]:
    #         all_cell_line_five_genes[key] = value
    # save_file = open('F:/DrugScreening2/try/myweb/cell_line_genes.pkl',"wb")
    # pickle.dump(all_cell_line_five_genes, save_file)
    # save_file.close()

# python -u webserver.py 'FC1=CNC(=O)NC1=O' 'FC1=CNC(=O)NC1=O' 'A2058' '1321985517@qq.com'



