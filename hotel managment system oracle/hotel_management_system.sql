
CREATE TABLE Rooms (
    RoomID NUMBER PRIMARY KEY,
    RoomType VARCHAR2(50),
    PricePerNight NUMBER(10, 2),
    AvailabilityStatus VARCHAR2(20)
);

CREATE TABLE Bookings (
    BookingID NUMBER PRIMARY KEY,
    CustomerID NUMBER REFERENCES Customers(CustomerID),
    RoomID NUMBER REFERENCES Rooms(RoomID),
    CheckInDate DATE,
    CheckOutDate DATE,
    TotalAmount NUMBER(10, 2)
);

CREATE TABLE Payments (
    PaymentID NUMBER PRIMARY KEY,
    BookingID NUMBER REFERENCES Bookings(BookingID),
    PaymentDate DATE,
    PaymentAmount NUMBER(10, 2),
    PaymentMethod VARCHAR2(50)
);

CREATE TABLE Staff (
    StaffID NUMBER PRIMARY KEY,
    FirstName VARCHAR2(50),
    LastName VARCHAR2(50),
    Role VARCHAR2(50),
    Salary NUMBER(10, 2),
    HireDate DATE
);

CREATE TABLE Services (
    ServiceID NUMBER PRIMARY KEY,
    ServiceName VARCHAR2(50),
    ServicePrice NUMBER(10, 2)
);

CREATE TABLE Service_Usage (
    UsageID NUMBER PRIMARY KEY,
    CustomerID NUMBER REFERENCES Customers(CustomerID),
    ServiceID NUMBER REFERENCES Services(ServiceID),
    UsageDate DATE,
    Quantity NUMBER
);


INSERT INTO Customers VALUES (1, 'John', 'Doe', '1234567890', 'john.doe@example.com', '123 Main St');
INSERT INTO Customers VALUES (2, 'Jane', 'Smith', '0987654321', 'jane.smith@example.com', '456 Elm St');

INSERT INTO Rooms VALUES (101, 'Single', 100, 'Available');
INSERT INTO Rooms VALUES (102, 'Double', 200, 'Available');
INSERT INTO Rooms VALUES (201, 'Suite', 500, 'Available');

INSERT INTO Bookings VALUES (1, 1, 101, TO_DATE('2024-12-01', 'YYYY-MM-DD'), TO_DATE('2024-12-05', 'YYYY-MM-DD'), 400);
INSERT INTO Bookings VALUES (2, 2, 201, TO_DATE('2024-12-03', 'YYYY-MM-DD'), TO_DATE('2024-12-07', 'YYYY-MM-DD'), 2000);

INSERT INTO Payments VALUES (1, 1, TO_DATE('2024-12-01', 'YYYY-MM-DD'), 400, 'Credit Card');
INSERT INTO Payments VALUES (2, 2, TO_DATE('2024-12-03', 'YYYY-MM-DD'), 2000, 'Cash');

INSERT INTO Staff VALUES (1, 'Alice', 'Brown', 'Manager', 5000, TO_DATE('2023-01-15', 'YYYY-MM-DD'));
INSERT INTO Staff VALUES (2, 'Bob', 'White', 'Receptionist', 3000, TO_DATE('2024-06-10', 'YYYY-MM-DD'));

INSERT INTO Services VALUES (1, 'Spa', 50);
INSERT INTO Services VALUES (2, 'Gym', 30);

INSERT INTO Service_Usage VALUES (1, 1, 1, TO_DATE('2024-12-02', 'YYYY-MM-DD'), 1);
INSERT INTO Service_Usage VALUES (2, 2, 2, TO_DATE('2024-12-04', 'YYYY-MM-DD'), 2);





