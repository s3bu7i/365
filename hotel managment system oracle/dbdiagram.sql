CREATE TABLE `Rooms` (
  `RoomID` int PRIMARY KEY,
  `RoomType` varchar(50),
  `PricePerNight` decimal(10,2),
  `AvailabilityStatus` varchar(20)
);

CREATE TABLE `Customers` (
  `CustomerID` int PRIMARY KEY,
  `FirstName` varchar(50),
  `LastName` varchar(50),
  `Phone` varchar(15),
  `Email` varchar(100),
  `Address` varchar(255)
);

CREATE TABLE `Bookings` (
  `BookingID` int PRIMARY KEY,
  `CustomerID` int,
  `RoomID` int,
  `CheckInDate` date,
  `CheckOutDate` date,
  `TotalAmount` decimal(10,2)
);

CREATE TABLE `Payments` (
  `PaymentID` int PRIMARY KEY,
  `BookingID` int,
  `PaymentDate` date,
  `PaymentAmount` decimal(10,2),
  `PaymentMethod` varchar(50)
);

CREATE TABLE `Staff` (
  `StaffID` int PRIMARY KEY,
  `FirstName` varchar(50),
  `LastName` varchar(50),
  `Role` varchar(50),
  `Salary` decimal(10,2),
  `HireDate` date
);

CREATE TABLE `Services` (
  `ServiceID` int PRIMARY KEY,
  `ServiceName` varchar(50),
  `ServicePrice` decimal(10,2)
);

CREATE TABLE `Service_Usage` (
  `UsageID` int PRIMARY KEY,
  `CustomerID` int,
  `ServiceID` int,
  `UsageDate` date,
  `Quantity` int
);

ALTER TABLE `Bookings` ADD FOREIGN KEY (`CustomerID`) REFERENCES `Customers` (`CustomerID`);

ALTER TABLE `Bookings` ADD FOREIGN KEY (`RoomID`) REFERENCES `Rooms` (`RoomID`);

ALTER TABLE `Payments` ADD FOREIGN KEY (`BookingID`) REFERENCES `Bookings` (`BookingID`);

ALTER TABLE `Service_Usage` ADD FOREIGN KEY (`CustomerID`) REFERENCES `Customers` (`CustomerID`);

ALTER TABLE `Service_Usage` ADD FOREIGN KEY (`ServiceID`) REFERENCES `Services` (`ServiceID`);
