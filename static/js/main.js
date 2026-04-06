document.getElementById("form").addEventListener("submit", async (e) => {
    e.preventDefault();

    document.getElementById("loading").classList.remove("hidden");

    const data = {
        ID: "1",
        Delivery_person_ID: "DP_1",
        Delivery_person_Age: document.getElementById("age").value,
        Delivery_person_Ratings: document.getElementById("ratings").value,
        Restaurant_latitude: 0,
        Restaurant_longitude: 0,
        Delivery_location_latitude: 0,
        Delivery_location_longitude: 0,
        Order_Date: "01-01-2024",
        Time_Orderd: "10:00",
        Time_Order_picked: "10:10",
        Weatherconditions: document.getElementById("weather").value,
        Road_traffic_density: document.getElementById("traffic").value,
        Vehicle_condition: 1,
        Type_of_order: "Snack",
        Type_of_vehicle: "Bike",
        multiple_deliveries: "0",
        Festival: "No",
        City: document.getElementById("city").value
    };

    const res = await fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    });

    const result = await res.json();

    document.getElementById("loading").classList.add("hidden");

    document.getElementById("result").innerHTML =
        "⏱ Delivery Time: " + result.prediction.toFixed(2) + " minutes";
});