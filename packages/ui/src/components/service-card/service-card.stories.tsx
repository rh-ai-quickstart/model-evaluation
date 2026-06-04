
import type { Meta, StoryObj } from '@storybook/react';
import { ServiceCard } from './service-card';
import { Server, Database, Monitor } from 'lucide-react';

const meta: Meta<typeof ServiceCard> = {
    title: 'Components/ServiceCard',
    component: ServiceCard,
    parameters: {
        layout: 'centered',
    },
    tags: ['autodocs'],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
    args: {
        name: 'Database',
        description: 'Stores and retrieves all application data.',
        icon: <Database className="h-5 w-5" />,
    },
};

export const AllPackages: Story = {
    render: () => (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-5xl">
            <ServiceCard name="UI" description="Frontend application interface." icon={<Monitor className="h-5 w-5" />} />
            <ServiceCard name="API" description="Handles all API requests." icon={<Server className="h-5 w-5" />} />
            <ServiceCard name="Database" description="Stores application data." icon={<Database className="h-5 w-5" />} />
        </div>
    ),
};
