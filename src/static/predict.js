  document.addEventListener('DOMContentLoaded', () => {

      document.querySelector('#form').onsubmit = () => {

          // Initialize new request
          const request = new XMLHttpRequest();
        //   const currency = document.querySelector('#currency').value;
          const radiant = [];
          const dire = []
          for (i = 0; i < 5; i++){
            radiant.push(document.querySelector(`#rHero${i}`).value);
            dire.push(document.querySelector(`#dHero${i}`).value);
          }
          request.open('POST', '/predict');

          // Callback function for when request completes
          request.onload = () => {

              // Extract JSON data from request
              const data = JSON.parse(request.responseText);

              // Update the result div
              const rforestpredictions =  `Scikit RForest Predicted Winrate:<br/>Radiant: ${data.rforestrate.toFixed(2)}% vs Dire: ${(100 - data.rforestrate).toFixed(2)}%`
              document.querySelector('#rforestrate').innerHTML = rforestpredictions;
              const nnmlppredictions =  `NNMLP Predicted Winrate:<br/>Radiant: ${data.nnmlprate.toFixed(2)}% vs Dire: ${(100 - data.nnmlprate).toFixed(2)}%`
              document.querySelector('#nnmlprate').innerHTML = nnmlppredictions;
              const xgbpredictions =  `XGBoost Predicted Winrate:<br/>Radiant: ${data.xgbrate.toFixed(2)}% vs Dire: ${(100 - data.xgbrate).toFixed(2)}%`
              document.querySelector('#xgbrate').innerHTML = xgbpredictions;
              const testpredictions =  `Quadratic Discriminant Predicted Winrate:<br/>Radiant: ${data.testrate.toFixed(2)}% vs Dire: ${(100 - data.testrate).toFixed(2)}%`
              document.querySelector('#testrate').innerHTML = testpredictions;
              const nn_predictions = `Deep Neural Network Predicted Winrate:<br/>Radiant: ${data.nn_rate.toFixed(2)}% vs Dire: ${(100 - data.nn_rate).toFixed(2)}%`
              document.querySelector('#nn_rate').innerHTML = nn_predictions;
              const avgpredictions = `Average Predicted Winrate:<br/>Radiant: ${data.avgrate.toFixed(2)}% vs Dire: ${(100 - data.avgrate).toFixed(2)}%`
              document.querySelector('#avgrate').innerHTML = avgpredictions;
          }

          // Add data to send with request
          const data = new FormData();
          data.append('rHero', radiant);
          data.append('dHero', dire);

          // Send request
          request.send(data);
          return false;
      };

  });
  