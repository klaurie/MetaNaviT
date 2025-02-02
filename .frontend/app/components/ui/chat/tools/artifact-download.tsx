"use client";

import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/app/components/ui/sheet";
import { Download, Eye, FileDown, Trash2, X } from "lucide-react";
import { useState, useEffect } from "react";
import { Button } from "../../button";
import { ScrollArea } from "../../scroll-area";
import { Artifact, CodeArtifact } from "./artifact";
import { cn } from "../../lib/utils";
import JSZip from "jszip";

interface ArtifactItem {
  id: string;
  artifact: CodeArtifact;
  timestamp: Date;
  version: number;
}

// Mock data for testing
const mockArtifacts: ArtifactItem[] = [
  {
    id: "1",
    artifact: {
      files: [
        {
          name: "example.py",
          content: `def hello_world():
    print("Hello, World!")
    return "Hello, World!"`,
          language: "python"
        }
      ]
    },
    timestamp: new Date(),
    version: 1
  }
];

export function ArtifactDownload() {
  const [artifacts, setArtifacts] = useState<ArtifactItem[]>(mockArtifacts);
  const [isOpen, setIsOpen] = useState(false);
  const [previewArtifact, setPreviewArtifact] = useState<CodeArtifact | null>(null);
  const [showPreview, setShowPreview] = useState(false);

  const onPreview = (item: ArtifactItem) => {
    setPreviewArtifact(item.artifact);
    setShowPreview(true);
  };

  const onDownload = async (item: ArtifactItem) => {
    try {
      const zip = new JSZip();
      
      item.artifact.files.forEach((file) => {
        zip.file(file.name, file.content);
      });
      
      const content = await zip.generateAsync({ type: "blob" });
      const url = window.URL.createObjectURL(content);
      const link = document.createElement('a');
      link.href = url;
      link.download = `artifact-${item.version}.zip`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  const onDelete = (item: ArtifactItem) => {
    setArtifacts(artifacts.filter((a) => a.id !== item.id));
  };

  return (
    <>
      <Sheet open={isOpen} onOpenChange={setIsOpen}>
        <SheetTrigger asChild>
          <Button variant="outline" className="fixed top-4 right-4">
            <FileDown className="h-4 w-4 mr-2" />
            Artifacts
          </Button>
        </SheetTrigger>
        <SheetContent>
          <SheetHeader>
            <SheetTitle>Generated Artifacts</SheetTitle>
          </SheetHeader>
          <ScrollArea className="h-[calc(100vh-8rem)] mt-4">
            <div className="space-y-4">
              {artifacts.length === 0 ? (
                <p className="text-center text-muted-foreground">
                  No artifacts generated yet
                </p>
              ) : (
                artifacts.map((item) => (
                  <ArtifactCard
                    key={item.id}
                    item={item}
                    onPreview={onPreview}
                    onDownload={onDownload}
                    onDelete={onDelete}
                  />
                ))
              )}
            </div>
          </ScrollArea>
        </SheetContent>
      </Sheet>

      {showPreview && previewArtifact && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50">
          <div className="fixed inset-x-4 top-4 bottom-4 bg-background rounded-lg shadow-lg border overflow-hidden">
            <div className="flex justify-between items-center p-4 border-b">
              <h3 className="text-lg font-semibold">Artifact Preview</h3>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setShowPreview(false)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
            <div className="p-4 h-[calc(100%-5rem)] overflow-auto">
              <Artifact artifact={previewArtifact} />
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function ArtifactCard({
  item,
  onPreview,
  onDownload,
  onDelete,
}: {
  item: ArtifactItem;
  onPreview: (item: ArtifactItem) => void;
  onDownload: (item: ArtifactItem) => void;
  onDelete: (item: ArtifactItem) => void;
}) {
  return (
    <div className="rounded-lg border p-4">
      <div className="flex items-center justify-between">
        <div className="space-y-1">
          <h4 className="text-sm font-medium">Artifact {item.version}</h4>
          <p className="text-sm text-muted-foreground">
            {item.artifact.files.length} file(s)
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            size="icon"
            variant="ghost"
            onClick={() => onPreview(item)}
            title="Preview"
          >
            <Eye className="h-4 w-4" />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            onClick={() => onDownload(item)}
            title="Download"
          >
            <Download className="h-4 w-4" />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            onClick={() => onDelete(item)}
            title="Delete"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
} 