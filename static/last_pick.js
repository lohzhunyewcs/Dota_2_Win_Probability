document.addEventListener('DOMContentLoaded', () => {
    let isRadiant = true;
    window.onload = () => {
        const team = document.querySelector('#team').innerHTML.trim();

        if (team ==='radiant'){
            document.getElementById('radiant4').remove();
        }
        else{
            document.getElementById('dire4').remove();
            isRadiant = false;
        }
        document.querySelector('#team').remove();
        document.querySelector('#form').onsubmit = () => {

            // Initialize new request
            const request = new XMLHttpRequest();
        //   const currency = document.querySelector('#currency').value;
            const radiant = [];
            const dire = []
            for (i = 0; i < 4; i++){
                radiant.push(document.querySelector(`#rHero${i}`).value);
                dire.push(document.querySelector(`#dHero${i}`).value);
            }
            if (!isRadiant){
                radiant.push(document.querySelector('#rHero4').value);
            }
            else{
                dire.push(document.querySelector('#dHero4').value);
            }
            request.open('POST', '/last_pick');

            // Callback function for when request completes
            request.onload = () => {

                // Extract JSON data from request
                const data = JSON.parse(request.responseText);

                // Update the result div
                // const predictions = `Predicted winrate:\nRadiant: ${data.rate.toFixed(2)}%\nDire: ${(100 - data.rate).toFixed(2)}%`
                // const predictions = `Recommended Hero: ${data.hero}\nPredicted winrate: ${data.rate.toFixed(2)}`
                let predictions = 'Top 5 Suggested Heroes:'
                for (i = 0; i < 5; i++){
                    predictions += `<br/>${i+1}. ${data.hero[i]}, ${data.rate[i].toFixed(2)}`
                }
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
    };

    

    

  });