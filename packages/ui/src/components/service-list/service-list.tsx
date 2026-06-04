
import { ServiceCard } from "../service-card/service-card";
import { Monitor, Server, Database } from "lucide-react";

const SERVICES = [
    { name: "UI", description: "Frontend application interface.", icon: <Monitor className="h-5 w-5" /> },
    { name: "API", description: "Handles all API requests and business logic.", icon: <Server className="h-5 w-5" /> },
    { name: "Database", description: "Stores and retrieves all application data.", icon: <Database className="h-5 w-5" /> },
];

export function ServiceList() {
    return (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {SERVICES.map((svc) => (
                <ServiceCard key={svc.name} name={svc.name} description={svc.description} icon={svc.icon} />
            ))}
        </div>
    );
}
