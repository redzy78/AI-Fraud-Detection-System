<?php
// Database connection details
$servername = "localhost";
$username = "root"; // Your MySQL username
$password = ""; // Your MySQL password
$dbname = "fraud"; // Your database name

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Remove fraud transaction based on TransactionID
if (isset($_POST['remove_fraud'])) {
    $transaction_id = $_POST['transaction_id']; 
    
    // Create the SQL query to delete the fraud entry
    $sql = "DELETE FROM transactions WHERE TransactionID = '$transaction_id'";

    if ($conn->query($sql) === TRUE) {
        echo "Transaction successfully removed!";
        header("Location: /results"); // Redirect to results page after removing
        exit();
    } else {
        echo "Error: " . $conn->error;
    }
}

// Re-run fraud detection process
if (isset($_POST['run_again'])) {
    // Fetch all transactions from the database
    $sql = "SELECT * FROM transactions";
    $result = $conn->query($sql);
    
    if ($result->num_rows > 0) {
        // Placeholder for your fraud detection logic
        echo "Fraud detection re-run successfully!";
    } else {
        echo "No transactions found!";
    }
}
?>
