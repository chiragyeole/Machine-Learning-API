echo '################Inserting the Glass indentification dataset, split it into training and test set and add the experiment in database####################'
echo
curl -X POST -H "Content-Type: application/json" -d '{
  "file": "glass.csv",
  "id": "1",
  "type": "classification"
}' http://localhost:5000/upload


echo
echo "##################Train the training set, save the model in Pickle and update the experiment with pickle file##################"
echo
curl -X POST -H "Content-Type: application/json" http://localhost:5000/train/1

echo
echo "##################Test the saved model on test set and update the experiment with accuracy and f1 score data###################"
echo
curl -X POST -H "Content-Type: application/json" http://localhost:5000/test/1

echo
echo "#################Predict for custom inserted data using the finalize trained model in pickle and print the output################"
echo
curl -X POST -H "Content-Type: application/json" -d '{
  "RI": "1.51348",
  "Na": "14.01",
  "Mg": "3.49",
  "Al": "1.71",
  "Si": "73.12",
  "K": "0.76",
  "Ca": "9.89",
  "Ba": "0.07",
  "Fe": "0.12"
}' http://localhost:5000/predict/1

echo
echo "##################Fetch the details of the experiment created earlier from database####################"
echo
curl localhost:5000/experiment/1

echo
echo "##############The following endpoint is just to check all the CRUD operation on the database are working fine###################"
echo
echo "################Create Experiment##################"
echo
curl -X POST -H "Content-Type: application/json" -d '{
  "id": "2",
  "name": "testing-record",
  "type": "classification",
  "train_data": "N/A",
  "test_data": "N/A",
  "result": "N/A",
  "model": "N/A"
}' http://localhost:5000/experiment

echo
echo "################Fetch all Experiments in the database###############"
echo
curl localhost:5000/experiment

echo
echo "#################Delete Experiment####################"
echo
curl -X "DELETE" localhost:5000/experiment/2

echo
echo "#######################################Working with another data set#############################################"
echo '###############Inserting the Breast Cancer classification dataset, split it into training and test set and add the experiment in database##################'
echo
curl -X POST -H "Content-Type: application/json" -d '{
  "file": "breast_cancer_coimbra.csv",
  "id": "3",
  "type": "classification"
}' http://localhost:5000/upload

echo
echo "#####################Train the training set, save the model in Pickle and update the experiment with pickle file#########################"
echo
curl -X POST -H "Content-Type: application/json" http://localhost:5000/train/3

echo
echo "#####################Test the saved model on test set and update the experiment with accuracy and f1 score data#######################"
echo
curl -X POST -H "Content-Type: application/json" http://localhost:5000/test/3

echo
echo "#####################Predict for custom inserted data using the finalized trained model in pickle and print the output#################"
echo
curl -X POST -H "Content-Type: application/json" -d '{
  "id": "3",
  "Age": "44",
  "BMI": "20.76",
  "Glucose": "86",
  "Insulin": "7.553",
  "HOMA": "1.6",
  "Leptin": "14.09",
  "Adiponectin": "20.32",
  "Resistin": "7.64",
  "MCP.1": "63.62"
}' http://localhost:5000/predict/3

echo
echo "#####################Fetch the details of the experiment created earlier from Database####################"
echo
curl localhost:5000/experiment/3
