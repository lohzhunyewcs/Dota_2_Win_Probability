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
              const predictions = `Predicted winrate:\nRadiant: ${data.rate.toFixed(2)}%\nDire: ${(100 - data.rate).toFixed(2)}%`
              document.querySelector('#result').innerHTML = predictions;
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