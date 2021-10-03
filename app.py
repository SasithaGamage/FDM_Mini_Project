from flask import Flask,request,render_template,send_file
import pickle
import numpy as np
import pandas as pd
import os


kmeans_model=pickle.load(open('FDM_kmeans.pkl','rb'))
knn_model=pickle.load(open('KNN.pkl','rb'))
decisionTree_model=pickle.load(open('Decision_Tree.pkl','rb'))
gradientBoost_model=pickle.load(open('Decision_Tree.pkl','rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/dataUpload')
def dataUpload():
    return render_template('CSVExport.html')

@app.route('/segmentPage')
def segmentPage():
    return render_template('segmentation.html')

@app.route('/segment',methods=['POST'])
def segment():
    result=request.form
    gender = result['gender']
    maritalStatus = result['maritalStatus']
    age = result['age']
    Graduated = result['graduated']
    profession =  result['profession']
    income = result['income']
    experience = result['experience']
    spendingScore = result['spendingScore']
    familySize = result['familySize']



    data = [[gender,maritalStatus,age,Graduated,profession,income,experience,spendingScore,familySize]]

    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns = ['Gender', 'Ever_Married','Age','Graduated','Profession','Income','Work_Experience','Spending_Score','Family_Size'])

    # -----------Encoding-------------------------------

    genderEncode = {'Male':0, 'Female':1}
    df.Gender = df.Gender.map(genderEncode)

    MarriedEncode = {'No':0, 'Yes':1}
    df.Ever_Married = df.Ever_Married.map(MarriedEncode)

    GraduatedEncode = {'No':0, 'Yes':1}
    df.Graduated = df.Graduated.map(GraduatedEncode)

    Spending_ScoreEncode = {'Low':0, 'Average':1, 'High':2}
    df.Spending_Score = df.Spending_Score.map(Spending_ScoreEncode)

    ProfessionEncode = {'Healthcare':1, 'Engineer':2, 'Lawyer':3, 'Entertainment':4, 'Artist':5, 'Executive':6, 'Doctor':7, 'Homemaker':8, 'Marketing':9}
    df.Profession = df.Profession.map(ProfessionEncode)

    test = df.to_numpy().reshape(1,-1)

    prediction = kmeans_model.predict(test)

    if (prediction == 0):
        customerSegment = "Luxury Vehicles"
    elif(prediction == 1):
        customerSegment = "Mid-Range Vehicles"
    elif(prediction == 2):
        customerSegment = "Family Vehicles"
    elif(prediction == 3):
        customerSegment = "Budget Vehicles"
    else:
        customerSegment = "No prediction"

    resultDict={"Prediction":customerSegment}
    return render_template('SegmentationResults.html',results = resultDict)


@app.route('/loan')
def loan():
    return render_template('loanPredict.html')

@app.route('/loanPredict',methods=['POST'])
def loanPredict():
    result=request.form


    predictionModel = result['predictionModel']
    income = result['income']
    age = result['age']
    experience = result['experience']
    maritalStatus =  result['maritalStatus']
    houseowner = result['houseowner']
    carowner = result['carowner']
    profession = result['profession']
    state = result['state']
    houseYears = result['houseYears']
    jobYears = result['jobYears']


    data = [[income,age,experience,maritalStatus,houseowner,carowner,profession,state,houseYears,jobYears]]

    df = pd.DataFrame(data, columns = ['Income', 'Age','Experience','Married_Single','House_Ownership','Car_Ownership','Profession','STATE','CURRENT_JOB_YRS','CURRENT_HOUSE_YRS'])

     # -----------Encoding-------------------------------

    Marital_Status_encode = {'single':0, 'married':1}

    df.Married_Single = df.Married_Single.map(Marital_Status_encode)

    House_Ownership_encode = {'rented':1, 'norent_noown':2, 'owned':3}
    df.House_Ownership = df.House_Ownership.map(House_Ownership_encode)

    Car_Ownership_encode = {'no':0, 'yes':1}
    df.Car_Ownership = df.Car_Ownership.map(Car_Ownership_encode)

    Profession_encode = {'Mechanical_engineer':1, 'Software_Developer':2, 'Technical_writer':3,'Civil_servant':4, 'Librarian':5, 'Economist':6, 'Flight_attendant':7,
                      'Architect':8, 'Designer':9, 'Physician':10, 'Financial_Analyst':11,'Air_traffic_controller':12, 'Politician':13, 'Police_officer':14, 'Artist':15,
                      'Surveyor':16, 'Design_Engineer':17, 'Chemical_engineer':18,'Hotel_Manager':19, 'Dentist':20, 'Comedian':21, 'Biomedical_Engineer':22,
                      'Graphic_Designer':23, 'Computer_hardware_engineer':24,'Petroleum_Engineer':25, 'Secretary':26, 'Computer_operator':27,
                      'Chartered_Accountant':28, 'Technician':29, 'Microbiologist':30,'Fashion_Designer':31, 'Aviator':32, 'Psychologist':33, 'Magistrate':34,
                      'Lawyer':35, 'Firefighter':36, 'Engineer':37, 'Official':38, 'Analyst':39,'Geologist':40, 'Drafter':41, 'Statistician':42, 'Web_designer':43,
                      'Consultant':44, 'Chef':45, 'Army_officer':46, 'Surgeon':47, 'Scientist':48,'Civil_engineer':49, 'Industrial_Engineer':50, 'Technology_specialist':51}
    df.Profession = df.Profession.map(Profession_encode)

    State_encode = {'Madhya_Pradesh':1, 'Maharashtra':2, 'Kerala':3, 'Odisha':4, 'Tamil_Nadu':5,'Gujarat':6, 'Rajasthan':7, 'Telangana':8, 'Bihar':9, 'Andhra_Pradesh':10,
                  'West_Bengal':11, 'Haryana':12, 'Puducherry':13, 'Karnataka':14,'Uttar_Pradesh':15, 'Himachal_Pradesh':16, 'Punjab':17, 'Tripura':18,
                  'Uttarakhand':19, 'Jharkhand':20, 'Mizoram':21, 'Assam':22,'Jammu_and_Kashmir':23, 'Delhi':24, 'Chhattisgarh':25, 'Chandigarh':26,
                  'Uttar_Pradesh[5]':27, 'Manipur':28, 'Sikkim':29}
    df.STATE = df.STATE.map(State_encode)

    test = df.to_numpy().reshape(1,-1)

    if predictionModel == 'knn':
        knnPredict = knn_model.predict(test)
        if knnPredict == 0:
            str = "Risky"
        else:
            str = "Not Risky"
        resultDict={"str":str,"Model":"K-Nearest-Neighbours"}
        return render_template('LoanPredictionResults.html',results = resultDict)
    elif(predictionModel == 'decisionTree'):
        decisionTreePredict = decisionTree_model.predict(test)
        if decisionTreePredict == 0:
            str = "Risky"
        else:
            str = "Not Risky"
        resultDict={"str":str,"Model":"Decision Tree"}
        return render_template('LoanPredictionResults.html',results = resultDict)
    else:
        gradientboostPredict = gradientBoost_model.predict(test)
        if gradientboostPredict == 0:
            str = "Risky"
        else:
            str = "Not Risky"
        resultDict={"str":str,"Model":"Gradient Boost"}
        return render_template('LoanPredictionResults.html',results = resultDict)

# pre processing function
def preProcess(data):
    data1 = data.drop(["ID","Var_1"],axis = 1)
    genderEncode = {'Male':0, 'Female':1}
    data1.Gender = data1.Gender.map(genderEncode)

    MarriedEncode = {'No':0, 'Yes':1}
    data1.Ever_Married = data1.Ever_Married.map(MarriedEncode)

    GraduatedEncode = {'No':0, 'Yes':1}
    data1.Graduated = data1.Graduated.map(GraduatedEncode)

    Spending_ScoreEncode = {'Low':1, 'Average':2, 'High':3}
    data1.Spending_Score = data1.Spending_Score.map(Spending_ScoreEncode)

    ProfessionEncode = {'Healthcare':1, 'Engineer':2, 'Lawyer':3, 'Entertainment':4, 'Artist':5, 'Executive':6, 'Doctor':7, 'Homemaker':8, 'Marketing':9}
    data1.Profession = data1.Profession.map(ProfessionEncode)
    data2=data1.dropna(axis = 0)

    return data2


# pre processing function dataexport2
def encoding(data1):

    genderEncode = {0:'Male', 1:'Female'}
    data1.Gender = data1.Gender.map(genderEncode)

    MarriedEncode = {0:'No', 1:'Yes'}
    data1.Ever_Married = data1.Ever_Married.map(MarriedEncode)

    GraduatedEncode = {0:'No', 1:'Yes'}
    data1.Graduated = data1.Graduated.map(GraduatedEncode)

    Spending_ScoreEncode = {1:'Low', 2:'Average', 3:'High'}
    data1.Spending_Score = data1.Spending_Score.map(Spending_ScoreEncode)

    ProfessionEncode = {1:'Healthcare', 2:'Engineer', 3:'Lawyer', 4:'Entertainment', 5:'Artist', 6:'Executive', 7:'Doctor', 8:'Homemaker', 9:'Marketing'}
    data1.Profession = data1.Profession.map(ProfessionEncode)

    Cluster = {0:'Luxury', 1:'family', 2:'Mid-Range',3:'Budget'}
    data1.cluster_number = data1.cluster_number.map(Cluster)


    return data1

#---------------------------------------------------

def createDf(df,prediction):
    df['cluster_number'] = prediction
    newDf = df
    return newDf


@app.route('/download')
def download_file():
	#path = "html2pdf.pdf"
	path = "output.csv"
	# path = "simple.docx"
	#path = "sample.txt"
	return send_file(path, as_attachment=True)

def addIDColumn(data,df1):
    data1 = data.drop(["Var_1"],axis = 1)
    data2=data1.dropna(axis = 0)
    data3 = data2['ID']
    df1.insert(0, 'ID', data3)
    return df1

@app.route('/dataImport',methods=['GET','POST'])
def dataImport():
    if request.method == 'POST':
        file = request.form['uploadFile']
        ext = file.split('.')[1]
        if ext == 'xlsx':
            data = pd.read_excel(file)
            test = preProcess(data)
            prediction = kmeans_model.predict(test)
            df = createDf(test,prediction)
            df1 = encoding(df)
            df2 = addIDColumn(data,df1)
            download_file()
            return render_template("segmentationcsvOutput.html",data = df2.to_html())
        elif (ext == 'csv'):
            data = pd.read_csv(file)
            test = preProcess(data)
            prediction = kmeans_model.predict(test)
            df = createDf(test,prediction)
            df1 = encoding(df)
            df2 = addIDColumn(data,df1)
            download_file()
            return render_template("segmentationcsvOutput.html",data = df2.to_html())
        else:
            return render_template("segmentationcsvOutput.html",data = "Unsupported File Type")




if __name__ == "__main__":
    app.run(debug = True)
