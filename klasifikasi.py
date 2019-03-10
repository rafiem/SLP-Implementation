#!/usr/bin/python3
import json
import csv
import math
import matplotlib.pyplot as plt

class SLP():
  def __init__(self, thetas, bias, alpha, filename, epoch, kfold):
    self.thetas                 = thetas
    self.bias                   = bias
    self.data                   = self.parsing_data(filename)
    self.alpha                  = alpha
    self.epoch                  = epoch
    self.kfold                  = kfold
    self.source_field           = ["x1","x2","x3","x4"]
    self.akurasi_train_epoch    = []
    self.loss_train_epoch       = []
    self.akurasi_validasi_epoch = []
    self.loss_validasi_epoch    = []


  def parsing_data(self, file_name):
    f = open(file_name)
    csv_obj = csv.DictReader(f, fieldnames = ( "x1","x2","x3","x4","name","type" ))
    parsed_data = eval(json.dumps([row for row in csv_obj]))[1:]
    f.close()
    return parsed_data


  def calculate_thetas(self, previous_thetas, previous_dthetas):
    return [(x + self.alpha*y) for x,y in zip(previous_thetas, previous_dthetas)]


  def calculate_bias(self, previous_bias, previous_dbias):
    return previous_bias+self.alpha*previous_dbias


  def result(self, flower_data, modified_theta, modified_bias):
    return sum(list(x*y for x,y in zip(flower_data,modified_theta))) + modified_bias


  def activation(self, modified_result):
    return 1/(1+math.exp(-modified_result))


  def error(self, flower_type, modified_activation):
    return pow(int(flower_type) - modified_activation, 2)


  def calculate_dthetas(self, flower_list, types, modified_activation):
    return [(2*flower*(int(types)-modified_activation)*(1-modified_activation)*modified_activation) for flower in flower_list]


  def calculate_dbias(self, types, modified_activation):
    return 2*(int(types)-modified_activation)*(1-modified_activation)*modified_activation


  def determine_loss(self, target, aktivasi):
    return pow(int(target)-aktivasi,2)/2


  def determine_accuration(self, activation):
    if activation > 0.5:
      return 1
    else:
      return 0

  # Fungsi untuk plotting data akurasi dan loss menjadi grafik
  def plotting(self):
    plt.figure(1)
    plt.plot(self.akurasi_train_epoch, color='red', label='Train')
    plt.plot(self.akurasi_validasi_epoch, color='blue', label='Validasi')
    plt.title('Accuracy, learning rate : {}'.format(self.alpha))
    plt.xlabel("Epoch")
    plt.ylabel("Avg Accuracy")
    plt.legend()

    plt.figure(2)
    plt.plot(self.loss_train_epoch, color='red', label='Train')
    plt.plot(self.loss_validasi_epoch, color='blue', label='Validasi')
    plt.title('Loss, learning rate : {}'.format(self.alpha))
    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss")
    plt.legend()

    plt.show()

  # Fungsi untuk mulai proses utama training dan validasi data
  # Hasil dari fungsi ini adalah me-set nilai rata-rata dari akurasi dan loss dari training dan validasi
  def start(self):
    data = [self.data[i:i+20] for i in range(0,100,+20)]

    for count in range(self.epoch):

      for proses in range(self.kfold):
        
        akurasi_train     = 0
        loss_train        = 0
        akurasi_validasi  = 0
        loss_validasi     = 0
        train_data        = []
        validasi_data     = []

        for i in range(len(data)):
          if i == proses:
            validasi_data += data[i]
          else:
            train_data    += data[i]
          
        hasil_training  = self.training(train_data, self.thetas, self.bias)
        self.thetas     = hasil_training["theta"]
        self.bias       = hasil_training["bias"]
        akurasi_train  += hasil_training["akurasi"]
        loss_train     += hasil_training["loss"]

        hasil_validasi     = self.validasi(validasi_data, self.thetas, self.bias)
        akurasi_validasi  += hasil_validasi["akurasi"]
        loss_validasi     += hasil_validasi["loss"]

      self.akurasi_train_epoch.append(akurasi_train/self.kfold)
      self.loss_train_epoch.append(loss_train/self.kfold)
      self.akurasi_validasi_epoch.append(akurasi_validasi/self.kfold)
      self.loss_validasi_epoch.append(loss_validasi/self.kfold)

  # Fungsi training menggunakan 4 segmen dari 100 data yang dibagi 5
  # Output yang dikeluarkan berupa theta dan bias yang di-update serta akurasi dan loss
  def training(self, training_data, theta, bias):
    akurasi = 0
    loss    = 0

    for item in range(len(training_data)):
      data_bunga = [float(training_data[item][self.source_field[x]]) for x in range(4)]

      result      = self.result(data_bunga, theta, bias)
      activation  = self.activation(result)
      error       = self.error(training_data[item]["type"], activation)
      dtheta      = self.calculate_dthetas(data_bunga, training_data[item]["type"], activation)
      dbias       = self.calculate_dbias(training_data[item]["type"], activation)
      theta       = self.calculate_thetas(theta, dtheta)
      bias        = self.calculate_bias(bias, dbias)

      if(int(training_data[item]["type"]) == self.determine_accuration(activation)):
        akurasi += 1
      loss += self.determine_loss(training_data[item]["type"], activation)

    return {"theta": theta, "bias": bias, "akurasi": akurasi/len(training_data), "loss": loss/len(training_data)}

  # Fungsi validasi menggunakan 1 segmen dari 100 data yang dibagi 5
  # Output yang dikeluarkan adalah akurasi dan loss 
  def validasi(self, validating_data, theta, bias):
    akurasi = 0
    loss    = 0

    for item in range(len(validating_data)):
      data_bunga  = [float(validating_data[item][self.source_field[x]]) for x in range(4)]

      result      = self.result(data_bunga, theta, bias)
      activation  = self.activation(result)

      if(int(validating_data[item]["type"]) == self.determine_accuration(activation)):
        akurasi += 1
      loss += self.determine_loss(validating_data[item]["type"], activation)
    
    return {"akurasi": akurasi/len(validating_data), "loss": loss/len(validating_data)}

    
# Main Program, Inisiasi nilai awal untuk Learning
if __name__ == '__main__':
  # Inisiasi nilai awal
  thetas      = [0.4, 0.6, 0.5, 0.3]
  bias        = 0.5
  alpha       = 0.8
  epoch       = 20
  kfold       = 5

  # Inisiasi awal class SLP dengan nilai yang sudah ditentukan
  klasifikasi = SLP(thetas, bias, alpha, "data_iris.csv", epoch, kfold)
  # Mulai start proses training dan validasi data
  klasifikasi.start()
  # Menampilkan grafik rata-rata akurasi dan loss dari training dan validasi
  klasifikasi.plotting()
