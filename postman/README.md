# AutoCare API Postman Collection

This directory contains Postman collections and environment files for testing and interacting with the AutoCare Backend API.

## Files

- `AutoCare_Backend_API.postman_collection.json`: The main Postman collection containing all API endpoints organized by category
- `AutoCare_API_Development.postman_environment.json`: Environment variables for local development
- `AutoCare_API_Production.postman_environment.json`: Environment variables for production use

## Setup Instructions

### Importing the Collection and Environment

1. Open Postman
2. Click on "Import" in the top left corner
3. Select all the JSON files in this directory
4. Click "Import" to add both the collection and environments to your Postman workspace

### Selecting the Environment

1. After importing, select the desired environment from the environment dropdown in the top right corner of Postman
2. Choose "AutoCare API - Development" for local development or "AutoCare API - Production" for production API

### Initial Setup

Before using the collection, you need to set some environment variables:

1. For development environment, most variables are pre-populated with test values
2. For production environment, you should update the following variables:
   - `user_email` - Your email address for authentication
   - `user_password` - Your password for authentication
   - Other fields will be automatically populated after successful authentication

### Authentication Flow

1. Execute the "Register User" request if you need to create a new account
2. Execute the "Login User" request to obtain authentication tokens
   - The collection automatically saves access and refresh tokens to environment variables
3. Subsequent requests will use the saved tokens for authentication

### Handling Token Expiration

If your access token expires:

1. Use the "Refresh Token" request to get a new access token using your refresh token
2. The collection automatically updates the access token in your environment variables

## Important Environment Variables

| Variable        | Description                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------- |
| `base_url`      | Base URL for the API (e.g., http://localhost:8000/v1 or https://autocare.yourdomain.com/v1) |
| `access_token`  | JWT access token obtained after login                                                       |
| `refresh_token` | JWT refresh token for obtaining new access tokens                                           |
| `user_id`       | ID of the authenticated user                                                                |
| `user_email`    | Email address used for authentication                                                       |
| `user_password` | Password used for authentication                                                            |
| `car_id`        | ID of the selected car for operations                                                       |
| `session_id`    | ID of the current chat session                                                              |

## Notes

- Sensitive information like tokens and passwords are stored as "secret" type variables in the environments
- You can modify any environment variable by clicking on the environment name in the top right and editing values
- The collection includes pre-request and test scripts to handle authentication token management automatically
