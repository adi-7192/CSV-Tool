"""
Column Mapping Service - Centralized column detection logic
"""
from typing import List, Optional
import pandas as pd

class ColumnMappingService:
    """
    Service to handle column name detection and standardization across the application.
    Centralizes the logic for finding specific columns in uploaded data.
    """
    
    @staticmethod
    def get_date_column(columns: List[str]) -> Optional[str]:
        """
        Find the date column from a list of columns.
        Checks common date column names in order of preference.
        """
        candidates = ['Invoice Date', 'invoice_date', 'order_date', 'Order Date', 'date', 'Date']
        for col in candidates:
            if col in columns:
                return col
        return None

    @staticmethod
    def get_transaction_type_column(columns: List[str]) -> Optional[str]:
        """
        Find the transaction type column.
        """
        candidates = ['Transaction Type', 'transaction_type', 'type', 'Type']
        for col in candidates:
            if col in columns:
                return col
        return None

    @staticmethod
    def get_revenue_column(columns: List[str]) -> Optional[str]:
        """
        Find the revenue column.
        Prioritizes 'revenue_calc' as it contains pre-calculated/cleaned values.
        """
        candidates = ['revenue_calc', 'revenue_amount', 'Invoice Amount', 'revenue_in_inr', 'amount', 'Amount']
        for col in candidates:
            if col in columns:
                return col
        return None

    @staticmethod
    def get_shipping_column(columns: List[str]) -> Optional[str]:
        """
        Find the shipping cost column.
        """
        candidates = ['shipping_loss_calc', 'shipping_amount', 'Shipping Amount', 'shipping_cost', 'Shipping Cost']
        for col in candidates:
            if col in columns:
                return col
        return None

    @staticmethod
    def get_order_id_column(columns: List[str]) -> Optional[str]:
        """
        Find the order ID column.
        """
        candidates = ['order_id', 'Invoice Number', 'Order ID', 'orderid']
        for col in candidates:
            if col in columns:
                return col
        return None

    @staticmethod
    def get_asin_column(columns: List[str]) -> Optional[str]:
        """
        Find the ASIN column.
        """
        candidates = ['asin', 'Asin', 'ASIN', 'Amazon ASIN']
        for col in candidates:
            if col in columns:
                return col
        return None

    @staticmethod
    def get_quantity_column(columns: List[str]) -> Optional[str]:
        """
        Find the quantity column.
        """
        candidates = ['quantity', 'Quantity', 'units_sold', 'Units Sold']
        for col in candidates:
            if col in columns:
                return col
        return None
    
    @staticmethod
    def get_city_column(columns: List[str]) -> Optional[str]:
        """
        Find the city/region column.
        """
        candidates = ['ship_city', 'Ship City', 'city', 'City', 'region', 'Region']
        for col in candidates:
            if col in columns:
                return col
        return None

    @staticmethod
    def normalize_city_name(city_name: str) -> Optional[str]:
        """
        Normalize city name using mapping and title case.
        Handles common variations and aliases for Indian cities.
        """
        if not city_name:
            return None
        
        # Handle pandas NaN values
        if pd.isna(city_name):
            return None
        
        city_str = str(city_name).strip().lower()
        if not city_str:
            return None
        
        # City normalization mapping
        city_mapping = {
            'bangalore': 'Bangalore',
            'bengaluru': 'Bangalore',
            'delhi': 'New Delhi',
            'new delhi': 'New Delhi',
            'delhi ncr': 'New Delhi',
            'bombay': 'Mumbai',
            'kolkata': 'Kolkata',
            'calcutta': 'Kolkata',
            'ahmedabad': 'Ahmedabad',
            'ahmadabad': 'Ahmedabad',
            'hyderabad': 'Hyderabad',
            'pune': 'Pune',
            'poona': 'Pune',
            'chennai': 'Chennai',
            'madras': 'Chennai',
            'gurugram': 'Gurugram',
            'gurgaon': 'Gurugram',
            'noida': 'Noida',
            'navi mumbai': 'Navi Mumbai',
            'new mumbai': 'Navi Mumbai',
            'thane': 'Thane',
        }
        
        # Check mapping first
        if city_str in city_mapping:
            return city_mapping[city_str]
        
        # Otherwise, apply title case
        words = city_str.split()
        normalized_words = [word.capitalize() for word in words]
        return ' '.join(normalized_words)
