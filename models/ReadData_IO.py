__author__ = 'jinghe'

'''
This file include classes and functions to read EHR data from database; process ICD-9 codes
output CSV results;
'''

"""import packages"""
import numpy as np
from datetime import datetime
import csv
import pickle


def getCode(element, CCS_dict):
    element = str(element)
    element = element.replace(".", "")
    element = element.capitalize()
    if CCS_dict.__contains__(element):
        dx = CCS_dict[element]
    elif len(element) < 5:
        element = element + '0'
        if CCS_dict.__contains__(element):
            dx = CCS_dict[element]
        else:
            element = element + '0'
            if CCS_dict.__contains__(element):
                dx = CCS_dict[element]
            else:
                dx = '0000 Others'
    elif not element[:-1].isdigit():
        element = element[:-1]
        if CCS_dict.__contains__(element):
            dx = CCS_dict[element]
        else:
            element = element + '0'
            if CCS_dict.__contains__(element):
                dx = CCS_dict[element]
            else:
                dx = '0000 Others'
    else:
        dx = '0000 Others'
    return dx


def CCs(CCStable):
    CCS = np.genfromtxt(CCStable, dtype=None, delimiter='\n')
    CCS = np.delete(CCS, [0, 1], 0)
    CCS_dict = {}
    for i in range(len(CCS)):
        if i % 2 == 0:
            value = CCS[i + 1].split(' ')
            for c in value:
                CCS_dict[c] = CCS[i]
    MHcodes = ['309.1', '309.24', '309.0', '309.3', '309.28', '309.4', '309.22',
               '309.21', '309.23', '309', '309.82', '309.2', '309.29', '309.83', '309.9',
               '309.89', '309.8', '309.81', '309.27', '293.84', '300.00', '300.0', '300',
               '300.02', '300.7', '300.3', '300.01', '300.23', '300.9', '300.20', '300.29',
               '300.39', '313.82', '300.5', '300.09', '308.9', '307.23', '300.15', '300.21',
               '308', '308.0', '308.3', '306.4', '296.26', '296.21', '296.22', '296.25',
               '296.23', '311', '296.3', '296.30', '296.2', '296.31', '296.32', '296.20', '296.90',
               '300.4', 'V62.84', '296.39', '296.30', '296.33', '296.34', '293.83']
    MHcodes = list(set(MHcodes))
    for i in MHcodes:
        CCS_dict[i.replace('.', '')] = 'Anxiety/Depression'
    dxs = np.unique(CCS_dict.values())
    return CCS_dict, dxs


class Patient:
    def __init__(self, name, age, gd, area):
        self.date_format = "%Y-%m-%d"
        self.ptid = name
        self.demo = [age, gd, area]
        self.visits = {}
        self.seqLen = 0
        self.dxs = {}
        self.md = 'N'
        self.seq = []

    def __getitem__(self, dt, dx):
        date = datetime.strptime(dt, self.date_format)
        if self.dxs.__contains__(dx):
            self.dxs[dx] += 1
        else:
            self.dxs[dx] = 1
        if not self.visits.__contains__(date):
            self.visits[date] = []
            self.visits[date].append(dx)
        else:
            self.visits[date].append(dx)
        return self.visits

    def getdxs(self):
        diags = []
        for d in self.visits.values():
            diags += d
        for v in set(diags):
            if self.dxs.__contains__(v):
                pass
            else:
                self.dxs[v] = 1
        return self.dxs


class MDPatient:
    def __init__(self, name, age, gd, area):
        self.date_format = "%Y-%m-%d"
        self.ptid = name
        self.demo = [age, gd, area]
        self.dxs = {}  # this contains all dxs before MD
        self.visits = {}  # this contains all the visits before MD
        self.status = 'withoutMD'
        self.MDdates = []
        self.gap = ''
        self.max_prior = ''

    def __getitem__(self, dt, dx):
        date = datetime.strptime(dt, self.date_format)
        if dx == 'Anxiety/Depression':
            self.MDdates.append(date)
            self.status = 'withMD'
            if self.visits.__contains__(date):
                del self.visits[date]
        elif self.status == 'withoutMD':
            if not self.visits.__contains__(date):
                self.visits[date] = []
            self.visits[date].append(dx)
        return self.visits, self.status

    def getdxsBeforeMD(self):
        diags = []
        for d in self.visits.values():
            diags += d
        for v in set(diags):
            if self.dxs.__contains__(v):
                pass
            else:
                self.dxs[v] = 1
        return self.dxs

    def getGap(self):
        visitdates = self.visits.keys()
        if len(visitdates) > 1:
            self.max_prior = max(self.visits.keys())
        else:
            self.max_prior = self.visits.keys()[0]
        if len(self.MDdates) > 1:
            self.firstMD = min(self.MDdates)
        else:
            self.firstMD = self.MDdates[0]
        self.gap = self.firstMD - self.max_prior
        return self.gap, self.max_prior, self.firstMD


def readDataMD2(filename, CCS_dict, selectedDXs):
    """Read data: non_mental disorder"""
    f = open(filename, 'rb')
    pts = {}
    for line in f.xreadlines():
        line = line.split(',')
        if len(line) > 9:
            line[1] = line[1] + line[2]
            line.pop(2)
        pid = str('pos' + line[0]) + '%#%' + str(line[1])
        code = getCode(line[3], CCS_dict)
        if code in selectedDXs:
            if not pts.__contains__(pid):
                pts[pid] = MDPatient(pid, line[4], line[5], line[8])
                pts[pid].__getitem__(line[2], code)
            else:
                pts[pid].__getitem__(line[2], code)
    for key, value in pts.items():
        value.getdxsBeforeMD()
    f.close()
    return pts


def readData2(filename, CCS_dict, selectedDXs):
    """Read data: non_mental disorder"""
    f = open(filename, 'rb')
    pts = {}
    data = []
    for line in f.xreadlines():
        line = line.split(',')
        if len(line) > 9:
            line[1] = line[1] + line[2]
            line.pop(2)
        data.append(line)
        pid = str('neg' + line[0]) + '%#%' + str(line[1])
        code = getCode(line[3], CCS_dict)
        if code in selectedDXs:
            if not pts.__contains__(pid):
                pts[pid] = Patient(pid, line[4], line[5], line[8])
                pts[pid].__getitem__(line[2], code)
            else:
                pts[pid].__getitem__(line[2], code)
    for key, value in pts.items():
        value.getdxs()
    f.close()
    return pts


def writetocsv(data, filename):
    f = open(filename, 'w', encoding='utf8')
    mywriter = csv.writer(f, delimiter='\t')
    for i in data:
        mywriter.writerow(i)
    f.close()
    print('Finish writing to csv!')


def writetocsv2(data, y, colnames, filename):
    f = open(filename, 'w', encoding='utf8')
    mywriter = csv.writer(f)
    colnames.append('response')
    colnames.insert(0, 'ptid')
    mywriter.writerow(colnames)
    for i, dt in enumerate(data):
        dt.append(y[i])
        mywriter.writerow(dt)
    f.close()
    print('Finish writing to csv!')


def readfromcsv(filename):
    f = open(filename, 'rb')
    myreader = csv.reader(f)
    results = []
    for row in myreader:
        row_v2 = []
        for ele in row:
            row_v2.append(float(ele))
        results.append(row_v2)
    return results


def readfrompickle(filename):
    #read
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results


