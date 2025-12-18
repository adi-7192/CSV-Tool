/**
 * Column Mapping Modal Component
 * 
 * Allows users to review and manually override column mappings before upload.
 * Shows required fields indicator and validates mappings.
 */

import React, { useState, useEffect } from 'react';
import { Modal, Table, Select, Alert, Button, Tag } from 'antd';
import { CheckCircleOutlined, ExclamationCircleOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Option } = Select;

export interface ColumnMapping {
  standardColumn: string;
  csvColumn: string | null;
  required: boolean;
  description?: string;
}

interface ColumnMappingModalProps {
  open: boolean;
  csvColumns: string[];
  detectedMapping: Record<string, string>;
  onConfirm: (mapping: Record<string, string>) => void;
  onCancel: () => void;
}

const REQUIRED_COLUMNS = ['order_id', 'revenue_amount'];
const COLUMN_DESCRIPTIONS: Record<string, string> = {
  order_id: 'Unique order identifier (e.g., Invoice Number, Order ID)',
  revenue_amount: 'Transaction amount (e.g., Invoice Amount, Total)',
  order_date: 'Order date (e.g., Invoice Date, Order Date)',
  sku: 'Product SKU (e.g., SKU, Product ID)',
  transaction_type: 'Transaction type (e.g., Type, Transaction Type)',
  quantity: 'Quantity sold (e.g., Qty, Quantity)',
};

const ColumnMappingModal: React.FC<ColumnMappingModalProps> = ({
  open,
  csvColumns,
  detectedMapping,
  onConfirm,
  onCancel,
}) => {
  const [mapping, setMapping] = useState<Record<string, string>>(detectedMapping);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  useEffect(() => {
    setMapping(detectedMapping);
  }, [detectedMapping]);

  // Build mapping data for table
  const mappingData: ColumnMapping[] = [
    { standardColumn: 'order_id', required: true, description: COLUMN_DESCRIPTIONS.order_id },
    { standardColumn: 'revenue_amount', required: true, description: COLUMN_DESCRIPTIONS.revenue_amount },
    { standardColumn: 'order_date', required: false, description: COLUMN_DESCRIPTIONS.order_date },
    { standardColumn: 'sku', required: false, description: COLUMN_DESCRIPTIONS.sku },
    { standardColumn: 'transaction_type', required: false, description: COLUMN_DESCRIPTIONS.transaction_type },
    { standardColumn: 'quantity', required: false, description: COLUMN_DESCRIPTIONS.quantity },
  ].map((item) => ({
    ...item,
    csvColumn: mapping[item.standardColumn] || null,
  }));

  // Validate mapping
  const validateMapping = (): boolean => {
    const errors: string[] = [];
    
    REQUIRED_COLUMNS.forEach((col) => {
      if (!mapping[col] || mapping[col].trim() === '') {
        errors.push(`${col} is required`);
      }
    });

    setValidationErrors(errors);
    return errors.length === 0;
  };

  const handleMappingChange = (standardColumn: string, csvColumn: string | null) => {
    const newMapping = { ...mapping };
    if (csvColumn) {
      newMapping[standardColumn] = csvColumn;
    } else {
      delete newMapping[standardColumn];
    }
    setMapping(newMapping);
    setValidationErrors([]);
  };

  const handleConfirm = () => {
    if (validateMapping()) {
      onConfirm(mapping);
    }
  };

  const columns: ColumnsType<ColumnMapping> = [
    {
      title: 'Required Field',
      dataIndex: 'required',
      key: 'required',
      width: 100,
      render: (required: boolean) => (
        required ? (
          <Tag color="red" icon={<ExclamationCircleOutlined />}>
            Required
          </Tag>
        ) : (
          <Tag color="default">Optional</Tag>
        )
      ),
    },
    {
      title: 'Standard Column',
      dataIndex: 'standardColumn',
      key: 'standardColumn',
      width: 180,
      render: (text: string, record: ColumnMapping) => (
        <div>
          <div style={{ fontWeight: 600 }}>{text}</div>
          {record.description && (
            <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
              {record.description}
            </div>
          )}
        </div>
      ),
    },
    {
      title: 'CSV Column',
      dataIndex: 'csvColumn',
      key: 'csvColumn',
      render: (csvColumn: string | null, record: ColumnMapping) => (
        <Select
          style={{ width: '100%' }}
          placeholder="Select CSV column"
          value={csvColumn}
          onChange={(value) => handleMappingChange(record.standardColumn, value)}
          allowClear
          showSearch
          filterOption={(input, option) => {
            const optionValue = option?.value || option?.label || '';
            return String(optionValue).toLowerCase().includes(input.toLowerCase());
          }}
        >
          {csvColumns.map((col) => (
            <Option key={col} value={col}>
              {col}
            </Option>
          ))}
        </Select>
      ),
    },
  ];

  const isValid = validationErrors.length === 0 && REQUIRED_COLUMNS.every((col) => mapping[col]);

  return (
    <Modal
      title="Review Column Mapping"
      open={open}
      onCancel={onCancel}
      width={800}
      footer={[
        <Button key="cancel" onClick={onCancel}>
          Cancel
        </Button>,
        <Button
          key="confirm"
          type="primary"
          onClick={handleConfirm}
          disabled={!isValid}
          icon={<CheckCircleOutlined />}
        >
          Confirm & Upload
        </Button>,
      ]}
    >
      <div style={{ marginBottom: '16px' }}>
        <Alert
          message="Map your CSV columns to standard fields"
          description="Select the CSV column that corresponds to each standard field. Required fields must be mapped to continue."
          type="info"
          showIcon
          style={{ marginBottom: '16px' }}
        />
      </div>

      {validationErrors.length > 0 && (
        <Alert
          message="Validation Errors"
          description={
            <ul style={{ margin: 0, paddingLeft: '20px' }}>
              {validationErrors.map((error, idx) => (
                <li key={idx}>{error}</li>
              ))}
            </ul>
          }
          type="error"
          showIcon
          style={{ marginBottom: '16px' }}
        />
      )}

      <Table
        columns={columns}
        dataSource={mappingData}
        rowKey="standardColumn"
        pagination={false}
        size="small"
      />

      <div style={{ marginTop: '16px', fontSize: '12px', color: '#666' }}>
        <strong>Tip:</strong> If a column is not found, you can leave it unmapped (optional fields only).
        Required fields must be mapped to proceed with upload.
      </div>
    </Modal>
  );
};

export default ColumnMappingModal;

