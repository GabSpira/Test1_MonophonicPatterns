
const firebaseConfig = {
    apiKey: "AIzaSyD7y9iBRvyxim9_9l8m5LQRexBdB63cmHo",
    authDomain: "maelisteningtest.firebaseapp.com",
    projectId: "maelisteningtest",
    storageBucket: "maelisteningtest.appspot.com",
    messagingSenderId: "953894482343",
    appId: "1:953894482343:web:a505e13240e8ea5dac0682",
    measurementId: "G-F2RS4PDMKP"
  };
  
// Inizializza l'app Firebase
firebase.initializeApp(firebaseConfig);


  // Test dummy di connessione a Firebase
if (firebase.apps.length === 0) {
    console.error("Modulo Firebase SDK non configurato correttamente!");
  } else {
    // console.log("Connessione a Firebase stabilita correttamente!");
  }

var db = firebase.firestore();

var results_1 = db.collection("Results_Test_1");
var results_2 = db.collection("Results_Test_2");
var results_3 = db.collection("Results_Test_3");

async function getTestVisitsCounts() {

  const querySnapshot1 = await results_1.get();
  const querySnapshot2 = await results_2.get();
  const querySnapshot3 = await results_3.get();

  // console.log(querySnapshot1);

  const count1 = querySnapshot1.size;
  const count2 = querySnapshot2.size;
  const count3 = querySnapshot3.size;

  // console.log(count1, count2, count3);

  // console.log('funziona');

  return {
    count1,
    count2,
    count3
  };

}

function sendUserDataToFirebase_test1(age, gender, years_training, country, feedback, rhythm, score) {
  
  var userData = {
    age: age,
    gender: gender,
    years_training: years_training,
    country: country,
    feedback: feedback,
  };

  var ratings = [];
  for (var i = 0; i < rhythm.length; i++) {
    var rating = {
      rhythm: rhythm[i],
      score: score[i]
    };
    ratings.push(rating);
  }

  var userRatings = ratings;

  console.log(userData, userRatings);

  // Aggiunta dei dati come documento nella raccolta creata

  results_1.add({
    userData: userData,
    userRatings: userRatings,
  })
  .then(function(docRef) {
    console.log("Documento aggiunto con ID:", docRef.id);
  })
  .catch(function(error) {
    console.error("Errore nell'aggiunta del documento:", error);
  });


}


function sendUserDataToFirebase_test2(age, gender, years_training, country, feedback, rhythm, score) {
  
  var userData = {
    age: age,
    gender: gender,
    years_training: years_training,
    country: country,
    feedback: feedback,
  };

  var ratings = [];
  for (var i = 0; i < rhythm.length; i++) {
    var rating = {
      rhythm: rhythm[i],
      score: score[i]
    };
    ratings.push(rating);
  }

  var userRatings = ratings;

  console.log(userData, userRatings);

  // Aggiunta dei dati come documento nella raccolta creata

  results_2.add({
    userData: userData,
    userRatings: userRatings,
  })
  .then(function(docRef) {
    console.log("Documento aggiunto con ID:", docRef.id);
  })
  .catch(function(error) {
    console.error("Errore nell'aggiunta del documento:", error);
  });

}

function sendUserDataToFirebase_test3(age, gender, years_training, country, feedback, rhythm, score) {
  
  var userData = {
    age: age,
    gender: gender,
    years_training: years_training,
    country: country,
    feedback: feedback,
  };

  var ratings = [];
  for (var i = 0; i < rhythm.length; i++) {
    var rating = {
      rhythm: rhythm[i],
      score: score[i]
    };
    ratings.push(rating);
  }

  var userRatings = ratings;

  console.log(userData, userRatings);

  // Aggiunta dei dati come documento nella raccolta creata

  results_3.add({
    userData: userData,
    userRatings: userRatings,
  })
  .then(function(docRef) {
    console.log("Documento aggiunto con ID:", docRef.id);
  })
  .catch(function(error) {
    console.error("Errore nell'aggiunta del documento:", error);
  });

}