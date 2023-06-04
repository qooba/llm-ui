$(document).ready(function() {
    const apiUrl = `${window.origin}/api/chat`; // Replace with your API URL

    function sendUserMessage() {
        const userMessage = $('#user-message').val();
        if (!userMessage.trim()) return;
        
        // Append user message to the chat
        $('#chat-container').append(`<div class="alert alert-warning"><strong>You:</strong> ${userMessage}</div>`);
        var el = $(`<div class="alert alert-success"></div>`).appendTo('#chat-container');

        // Call API and process the response
        callApi(userMessage, el).catch(error => {
            console.error('Error fetching text stream:', error);
            el.text('Error fetching text stream.');
        });

        $('#user-message').val('');
    }

    async function callApi(message, el) {
        const response = await fetch(`${apiUrl}?prompt=${message}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            },
        });

        if (!response.ok) {
            throw new Error(`Error fetching stream: ${response.statusText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');

        while (true) {
            const { value, done } = await reader.read();
            if (done) {
                break;
            }

            const textChunk = decoder.decode(value, { stream: true });
            el.text(el.text() + textChunk);
        }
    }

    $('#send-message').on('click', sendUserMessage);
    $('#user-message').on('keypress', function(e) {
        if (e.which === 13) {
            sendUserMessage();
        }
    });
});
